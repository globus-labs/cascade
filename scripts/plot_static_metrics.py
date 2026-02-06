#!/usr/bin/env python3
"""
Static visualization of FD counts and trajectory distribution over time.

Creates two static plots:
1. File descriptor counts over time (total, pipes, files, sockets)
2. Number of trajectories in each agent over time
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Log patterns to extract trajectory state transitions
LOG_PATTERNS = {
    'dynamics_received': re.compile(
        r'Received advance spec for traj (\d+) chunk (\d+)'
    ),
    'dynamics_submit_audit': re.compile(
        r'Submitting audit for chunk (\d+) of traj (\d+)'
    ),
    'auditor_received': re.compile(
        r'Received chunk (\d+) from traj (\d+)'
    ),
    'auditor_submit_task': re.compile(
        r'Submitting audit of chunk (\d+) of traj (\d+) to executor'
    ),
    'audit_passed': re.compile(
        r'Audit passed for chunk (\d+) of traj (\d+)'
    ),
    'audit_failed': re.compile(
        r'Audit failed for chunk (\d+) of traj (\d+)'
    ),
    'next_chunk_submitted': re.compile(
        r'Submitted next chunk (\d+) for traj (\d+)'
    ),
    'auditor_submit_sampler': re.compile(
        r'Submitting failed chunk (\d+) of traj (\d+) to sampler'
    ),
    'sampler_received': re.compile(
        r'Sampling frames from chunk (\d+) of traj (\d+)'
    ),
    'sampler_submit_labeler': re.compile(
        r'Submitting training frame from traj (\d+) chunk (\d+) to labeler'
    ),
    'labeler_received': re.compile(
        r'Received training frame \(trajectory_frame_id=\d+\)'
    ),
    'trajectory_complete': re.compile(
        r'Traj (\d+) is complete'
    ),
    'submitted_to_dynamics': re.compile(
        r'submitted.*traj[ectory]*\s*(\d+)',
        re.IGNORECASE
    ),
    'submitting_to_audit': re.compile(
        r'submitting.*traj[ectory]*\s*(\d+).*chunk\s*(\d+).*audit',
        re.IGNORECASE
    ),
}

# FD count patterns - separate patterns for distinct log lines
FD_TOTAL_PATTERN = re.compile(
    r'FD total count: (\d+)'
)
FD_BREAKDOWN_PATTERN = re.compile(
    r'FD breakdown: pipes=(\d+), files=(\d+), sockets=(\d+)'
)
FD_WARNING_PATTERN = re.compile(
    r'FD WARNING: (.+)'
)
# Legacy pattern for backward compatibility
FD_LEGACY_PATTERN = re.compile(
    r'FD counts: total=(\d+), pipes=(\d+), files=(\d+), sockets=(\d+)'
)

# Resource counts pattern
RESOURCE_PATTERN = re.compile(
    r'Resource counts: agents=(\d+), db_connections=(\d+), traj_db_instances=(\d+)'
)

# SQLAlchemy pool stats pattern (handles optional fields)
POOL_STATS_PATTERN = re.compile(
    r'SQLAlchemy pool stats: size=(\d+), checked_in=(\d+), checked_out=(\d+), overflow=(\d+)(?:, invalid=(\d+))?'
)

# Executor worker health patterns - extract individual values (dict keys can be in any order)
EXECUTOR_HEALTH_ALIVE = re.compile(r'\'alive\'\s*:\s*(\d+)')
EXECUTOR_HEALTH_DEAD = re.compile(r'\'dead\'\s*:\s*(\d+)')
EXECUTOR_HEALTH_ZOMBIES = re.compile(r'\'zombies\'\s*:\s*(\d+)')
EXECUTOR_HEALTH_TOTAL = re.compile(r'\'total_workers\'\s*:\s*(\d+)')


def parse_log_file(log_file: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parse log file to extract trajectory events, FD counts, resource counts, pool stats, and executor health."""
    trajectory_events = []
    fd_counts = []
    resource_counts = []
    pool_stats = []
    executor_health = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # Extract timestamp
            timestamp_match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\]', line)
            if not timestamp_match:
                continue
            
            timestamp_str = timestamp_match.group(1)
            try:
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
            except ValueError:
                continue
            
            # Extract agent name
            agent_match = re.search(r'\((\w+)\)', line)
            agent = agent_match.group(1) if agent_match else 'Unknown'
            
            # Check for trajectory events
            for event_type, pattern in LOG_PATTERNS.items():
                match = pattern.search(line)
                if match:
                    if event_type in ['dynamics_received', 'dynamics_submit_audit', 'next_chunk_submitted', 'submitting_to_audit']:
                        if len(match.groups()) >= 2:
                            traj_id, chunk_id = int(match.group(1)), int(match.group(2))
                        else:
                            traj_id = int(match.group(1))
                            chunk_id = None
                        trajectory_events.append({
                            'timestamp': timestamp,
                            'agent': agent,
                            'event_type': event_type,
                            'traj_id': traj_id,
                            'chunk_id': chunk_id,
                        })
                    elif event_type in ['auditor_received', 'auditor_submit_task', 'audit_passed', 'audit_failed', 
                                        'auditor_submit_sampler', 'sampler_received', 'sampler_submit_labeler']:
                        if len(match.groups()) >= 2:
                            chunk_id, traj_id = int(match.group(1)), int(match.group(2))
                        else:
                            traj_id = int(match.group(1))
                            chunk_id = None
                        trajectory_events.append({
                            'timestamp': timestamp,
                            'agent': agent,
                            'event_type': event_type,
                            'traj_id': traj_id,
                            'chunk_id': chunk_id,
                        })
                    elif event_type == 'labeler_received':
                        # Labeler received doesn't have traj_id in the message, skip for now
                        # or we could try to extract from context, but for now skip
                        pass
                    elif event_type in ['trajectory_complete', 'submitted_to_dynamics']:
                        traj_id = int(match.group(1))
                        trajectory_events.append({
                            'timestamp': timestamp,
                            'agent': agent,
                            'event_type': event_type,
                            'traj_id': traj_id,
                            'chunk_id': None,
                        })
                    break
            
            # Check for FD counts - handle new distinct log lines
            fd_total_match = FD_TOTAL_PATTERN.search(line)
            fd_breakdown_match = FD_BREAKDOWN_PATTERN.search(line)
            fd_legacy_match = FD_LEGACY_PATTERN.search(line)
            
            # Handle new format: separate total and breakdown lines
            if fd_total_match:
                total = int(fd_total_match.group(1))
                # Look for breakdown on next lines (within 1 second)
                # For now, just record total - we'll merge breakdowns later
                fd_counts.append({
                    'timestamp': timestamp,
                    'total': total,
                    'pipes': None,
                    'files': None,
                    'sockets': None,
                })
            elif fd_breakdown_match:
                # Breakdown line - try to match with most recent total
                pipes = int(fd_breakdown_match.group(1))
                files = int(fd_breakdown_match.group(2))
                sockets = int(fd_breakdown_match.group(3))
                
                # Find most recent total entry without breakdown and update it
                for fd_entry in reversed(fd_counts):
                    if fd_entry['pipes'] is None and fd_entry['total'] is not None:
                        # Update this entry with breakdown
                        fd_entry['pipes'] = pipes
                        fd_entry['files'] = files
                        fd_entry['sockets'] = sockets
                        break
                else:
                    # No matching total found, create new entry with just breakdown
                    fd_counts.append({
                        'timestamp': timestamp,
                        'total': None,
                        'pipes': pipes,
                        'files': files,
                        'sockets': sockets,
                    })
            elif fd_legacy_match:
                # Legacy format - all in one line
                fd_counts.append({
                    'timestamp': timestamp,
                    'total': int(fd_legacy_match.group(1)),
                    'pipes': int(fd_legacy_match.group(2)),
                    'files': int(fd_legacy_match.group(3)),
                    'sockets': int(fd_legacy_match.group(4)),
                })
            
            # Check for FD warnings
            fd_warning_match = FD_WARNING_PATTERN.search(line)
            if fd_warning_match:
                # Store warnings separately for potential analysis
                # Could add a warnings list if needed
                pass
            
            # Check for resource counts
            resource_match = RESOURCE_PATTERN.search(line)
            if resource_match:
                resource_counts.append({
                    'timestamp': timestamp,
                    'agents': int(resource_match.group(1)),
                    'db_connections': int(resource_match.group(2)),
                    'traj_db_instances': int(resource_match.group(3)),
                })
            
            # Check for pool stats
            pool_match = POOL_STATS_PATTERN.search(line)
            if pool_match:
                pool_stats.append({
                    'timestamp': timestamp,
                    'size': int(pool_match.group(1)),
                    'checked_in': int(pool_match.group(2)),
                    'checked_out': int(pool_match.group(3)),
                    'overflow': int(pool_match.group(4)),
                })
            
            # Check for executor health (only if not error/skipped)
            if ('Executor worker health:' in line or 'check result:' in line) and 'error' not in line.lower() and 'skipped' not in line.lower():
                # Extract individual values (dict keys can be in any order)
                alive_match = EXECUTOR_HEALTH_ALIVE.search(line)
                dead_match = EXECUTOR_HEALTH_DEAD.search(line)
                zombies_match = EXECUTOR_HEALTH_ZOMBIES.search(line)
                total_match = EXECUTOR_HEALTH_TOTAL.search(line)
                
                if alive_match and dead_match and zombies_match and total_match:
                    executor_health.append({
                        'timestamp': timestamp,
                        'agent': agent,
                        'alive': int(alive_match.group(1)),
                        'dead': int(dead_match.group(1)),
                        'zombies': int(zombies_match.group(1)),
                        'total_workers': int(total_match.group(1)),
                    })
    
    trajectory_df = pd.DataFrame(trajectory_events)
    fd_df = pd.DataFrame(fd_counts)
    resource_df = pd.DataFrame(resource_counts)
    pool_df = pd.DataFrame(pool_stats)
    executor_df = pd.DataFrame(executor_health)
    
    # Process FD dataframe to handle None values
    if not fd_df.empty:
        # Sort by timestamp
        fd_df = fd_df.sort_values('timestamp').reset_index(drop=True)
        
        # Forward-fill breakdown values (pipes, files, sockets) from later entries
        # Backward-fill total values from earlier entries
        # This handles cases where total and breakdown are on separate log lines
        # Since they're logged close together, this should work well
        
        # Use pandas fillna with method parameter (deprecated but still works)
        # For newer pandas, we'd use ffill() and bfill() methods
        try:
            # Try new pandas API first
            fd_df['pipes'] = fd_df['pipes'].ffill()
            fd_df['files'] = fd_df['files'].ffill()
            fd_df['sockets'] = fd_df['sockets'].ffill()
            fd_df['total'] = fd_df['total'].bfill()
        except AttributeError:
            # Fall back to old API
            fd_df['pipes'] = fd_df['pipes'].fillna(method='ffill')
            fd_df['files'] = fd_df['files'].fillna(method='ffill')
            fd_df['sockets'] = fd_df['sockets'].fillna(method='ffill')
            fd_df['total'] = fd_df['total'].fillna(method='bfill')
        
        # Fill any remaining None values with 0
        fd_df = fd_df.fillna(0)
        
        # Convert to int
        for col in ['total', 'pipes', 'files', 'sockets']:
            if col in fd_df.columns:
                fd_df[col] = fd_df[col].astype(int)
    
    return trajectory_df, fd_df, resource_df, pool_df, executor_df


def track_trajectory_states(trajectory_df: pd.DataFrame) -> Dict[int, List[Tuple[datetime, str]]]:
    """Track which agent has each trajectory at each point in time."""
    traj_states = defaultdict(list)
    
    for traj_id in trajectory_df['traj_id'].unique():
        traj_events = trajectory_df[trajectory_df['traj_id'] == traj_id].sort_values('timestamp')
        
        current_agent = None
        for _, event in traj_events.iterrows():
            event_type = event['event_type']
            timestamp = event['timestamp']
            
            # Determine new agent state based on event type
            new_agent = None
            
            # State transitions based on event type
            # DynamicsEngine events
            if event_type in ['dynamics_received', 'submitted_to_dynamics', 'next_chunk_submitted']:
                new_agent = 'DynamicsEngine'
            # Auditor events - transition to auditor when chunk arrives or is submitted
            elif event_type in ['dynamics_submit_audit', 'submitting_to_audit', 'auditor_received', 'auditor_submit_task']:
                new_agent = 'Auditor'
            # Audit result - passed means next chunk goes to dynamics, failed goes to sampler
            elif event_type == 'audit_passed':
                # Check if there's a next chunk coming
                future_events = traj_events[traj_events['timestamp'] > timestamp]
                if any(future_events['event_type'] == 'next_chunk_submitted'):
                    # Next chunk will be submitted, so trajectory stays with auditor briefly
                    # then transitions to dynamics when next_chunk_submitted is seen
                    new_agent = 'Auditor'
                else:
                    new_agent = None  # Trajectory complete
            elif event_type in ['audit_failed', 'auditor_submit_sampler']:
                # Failed audit - trajectory transitions to sampler
                new_agent = 'DummySampler'
            # Sampler events
            elif event_type in ['sampler_received', 'sampler_submit_labeler']:
                new_agent = 'DummySampler'
            # Labeler events
            elif event_type == 'labeler_received':
                new_agent = 'DummyLabeler'
            # Completion
            elif event_type == 'trajectory_complete':
                new_agent = None
            
            # Only add state entry if agent actually changed
            # This prevents duplicate entries when multiple events occur for the same agent
            if new_agent != current_agent:
                current_agent = new_agent
                if current_agent:
                    traj_states[traj_id].append((timestamp, current_agent))
                # If new_agent is None (trajectory complete), we don't add an entry
                # The trajectory will remain in its last state until completion
    
    return dict(traj_states)


def get_trajectory_at_time(
    traj_states: Dict[int, List[Tuple[datetime, str]]],
    time: datetime
) -> Dict[str, List[int]]:
    """Get which trajectories are at each agent at a given time."""
    result = defaultdict(list)
    for traj_id, states in traj_states.items():
        # Find the most recent state before or at this time
        agent = None
        for state_time, state_agent in states:
            if state_time <= time:
                agent = state_agent
            else:
                break
        if agent:
            result[agent].append(traj_id)
    return dict(result)


def create_static_plots(
    trajectory_df: pd.DataFrame,
    fd_df: pd.DataFrame,
    resource_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    executor_df: pd.DataFrame
) -> go.Figure:
    """Create static plots: FD counts, trajectory counts per agent, database connection metrics, and executor health."""
    # Create subplots: FD plot on top, trajectory counts, DB connections, executor health on bottom
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            'File Descriptor Counts Over Time',
            'Trajectory Counts per Agent Over Time',
            'Database Connection Metrics Over Time',
            'Executor Worker Process Health by Agent'
        ),
        vertical_spacing=0.10,
        row_heights=[0.25, 0.25, 0.25, 0.25]
    )
    
    # List of all agents
    all_agents = ['DynamicsEngine', 'Auditor', 'DummySampler', 'DummyLabeler', 'DummyTrainer', 'DatabaseMonitor']
    
    # Track trajectory states
    traj_states = {}
    if not trajectory_df.empty:
        traj_states = track_trajectory_states(trajectory_df)
    
    # Plot 1: FD counts over time
    if not fd_df.empty:
        fd_df = fd_df.sort_values('timestamp')
        fig.add_trace(
            go.Scatter(
                x=fd_df['timestamp'],
                y=fd_df['total'],
                mode='lines+markers',
                name='Total FDs',
                line=dict(color='red', width=2),
                showlegend=True,
                legendgroup='fd',
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=fd_df['timestamp'],
                y=fd_df['pipes'],
                mode='lines',
                name='Pipes',
                line=dict(color='blue', width=1.5),
                showlegend=True,
                legendgroup='fd',
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=fd_df['timestamp'],
                y=fd_df['files'],
                mode='lines',
                name='Files',
                line=dict(color='green', width=1.5),
                showlegend=True,
                legendgroup='fd',
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=fd_df['timestamp'],
                y=fd_df['sockets'],
                mode='lines',
                name='Sockets',
                line=dict(color='orange', width=1.5),
                showlegend=True,
                legendgroup='fd',
            ),
            row=1, col=1
        )
    
    # Plot 2: Trajectory counts per agent over time
    if traj_states:
        # Determine time range (include all data sources)
        time_sources = [trajectory_df['timestamp']]
        if not fd_df.empty:
            time_sources.append(fd_df['timestamp'])
        if not resource_df.empty:
            time_sources.append(resource_df['timestamp'])
        if not pool_df.empty:
            time_sources.append(pool_df['timestamp'])
        if not executor_df.empty:
            time_sources.append(executor_df['timestamp'])
        
        min_time = min(ts.min() for ts in time_sources)
        max_time = max(ts.max() for ts in time_sources)
        
        # Create time steps (every 1 second for smoother plots)
        time_steps = pd.date_range(start=min_time, end=max_time, freq='1S')
        
        # Build data for each agent
        agent_data = {agent: {'times': [], 'counts': []} for agent in all_agents}
        
        for t in time_steps:
            trajs_by_agent = get_trajectory_at_time(traj_states, t)
            for agent in all_agents:
                count = len(trajs_by_agent.get(agent, []))
                agent_data[agent]['times'].append(t)
                agent_data[agent]['counts'].append(count)
        
        # Plot each agent
        colors = {
            'DynamicsEngine': 'blue',
            'Auditor': 'green',
            'DummySampler': 'orange',
            'DummyLabeler': 'purple',
            'DummyTrainer': 'brown',
            'DatabaseMonitor': 'pink',
        }
        
        for agent in all_agents:
            if agent_data[agent]['times']:
                fig.add_trace(
                    go.Scatter(
                        x=agent_data[agent]['times'],
                        y=agent_data[agent]['counts'],
                        mode='lines+markers',
                        name=agent,
                        line=dict(color=colors.get(agent, 'gray'), width=2),
                        marker=dict(size=4),
                        showlegend=True,
                        legendgroup='agents',
                    ),
                    row=2, col=1
                )
    
    # Plot 3: Database connection metrics
    if not resource_df.empty:
        resource_df = resource_df.sort_values('timestamp')
        fig.add_trace(
            go.Scatter(
                x=resource_df['timestamp'],
                y=resource_df['db_connections'],
                mode='lines+markers',
                name='DB Connections',
                line=dict(color='purple', width=2),
                showlegend=True,
                legendgroup='db',
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=resource_df['timestamp'],
                y=resource_df['traj_db_instances'],
                mode='lines+markers',
                name='TrajDB Instances',
                line=dict(color='brown', width=2),
                showlegend=True,
                legendgroup='db',
            ),
            row=3, col=1
        )
    
    if not pool_df.empty:
        pool_df = pool_df.sort_values('timestamp')
        fig.add_trace(
            go.Scatter(
                x=pool_df['timestamp'],
                y=pool_df['size'],
                mode='lines+markers',
                name='Pool Size',
                line=dict(color='red', width=2),
                showlegend=True,
                legendgroup='pool',
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=pool_df['timestamp'],
                y=pool_df['checked_out'],
                mode='lines',
                name='Checked Out',
                line=dict(color='orange', width=1.5),
                showlegend=True,
                legendgroup='pool',
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=pool_df['timestamp'],
                y=pool_df['checked_in'],
                mode='lines',
                name='Checked In',
                line=dict(color='green', width=1.5),
                showlegend=True,
                legendgroup='pool',
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=pool_df['timestamp'],
                y=pool_df['overflow'],
                mode='lines',
                name='Overflow',
                line=dict(color='blue', width=1.5),
                showlegend=True,
                legendgroup='pool',
            ),
            row=3, col=1
        )
    
    # Plot 4: Executor worker process health by agent
    if not executor_df.empty:
        executor_df = executor_df.sort_values('timestamp')
        # Get unique agents that have executor health data
        agents_with_executors = executor_df['agent'].unique()
        
        # Colors for process states
        for agent in agents_with_executors:
            agent_data = executor_df[executor_df['agent'] == agent]
            
            # Alive workers
            fig.add_trace(
                go.Scatter(
                    x=agent_data['timestamp'],
                    y=agent_data['alive'],
                    mode='lines+markers',
                    name=f'{agent} - Alive',
                    line=dict(color='green', width=2),
                    showlegend=True,
                    legendgroup=f'exec_{agent}',
                ),
                row=4, col=1
            )
            
            # Dead workers
            fig.add_trace(
                go.Scatter(
                    x=agent_data['timestamp'],
                    y=agent_data['dead'],
                    mode='lines+markers',
                    name=f'{agent} - Dead',
                    line=dict(color='red', width=2),
                    showlegend=True,
                    legendgroup=f'exec_{agent}',
                ),
                row=4, col=1
            )
            
            # Zombie workers
            fig.add_trace(
                go.Scatter(
                    x=agent_data['timestamp'],
                    y=agent_data['zombies'],
                    mode='lines+markers',
                    name=f'{agent} - Zombies',
                    line=dict(color='orange', width=2),
                    showlegend=True,
                    legendgroup=f'exec_{agent}',
                ),
                row=4, col=1
            )
            
            # Total workers
            fig.add_trace(
                go.Scatter(
                    x=agent_data['timestamp'],
                    y=agent_data['total_workers'],
                    mode='lines',
                    name=f'{agent} - Total',
                    line=dict(color='blue', width=1.5, dash='dash'),
                    showlegend=True,
                    legendgroup=f'exec_{agent}',
                ),
                row=4, col=1
            )
    
    # Update layout with synchronized hover
    fig.update_layout(
        title='FD Counts, Trajectory Distribution, Database Connection Metrics, and Executor Health Over Time',
        height=1600,
        xaxis_title='Time',
        yaxis_title='FD Count',
        xaxis2_title='Time',
        yaxis2_title='Number of Trajectories',
        xaxis3_title='Time',
        yaxis3_title='Connection Count',
        xaxis4_title='Time',
        yaxis4_title='Number of Workers',
        hovermode='x unified',  # This shows hover info for all traces at same x
    )
    
    # Link x-axes for synchronized hover, zoom, and pan
    # This makes hovering on one plot show info on all three
    fig.update_xaxes(
        matches='x',  # Link x-axes
        showspikes=True,  # Show vertical line on hover
        spikecolor='gray',
        spikethickness=1,
        spikemode='across',  # Show spike across all subplots
        spikedash='dot',
    )
    
    # Position legend to show all groups with gaps between them
    fig.update_layout(
        legend=dict(
            orientation='v',
            yanchor='top',
            y=0.98,
            xanchor='left',
            x=1.02,
            tracegroupgap=20,  # Gap between groups
        ),
    )
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Create static plots of FD counts and trajectory distribution over time'
    )
    parser.add_argument(
        'log_file',
        type=Path,
        help='Path to log file to parse'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output HTML file (default: log_file with _static.html suffix)'
    )
    
    args = parser.parse_args()
    
    if not args.log_file.exists():
        print(f"Error: Log file {args.log_file} does not exist")
        return 1
    
    print(f"Parsing log file: {args.log_file}")
    trajectory_df, fd_df, resource_df, pool_df, executor_df = parse_log_file(args.log_file)
    
    print(f"Found {len(trajectory_df)} trajectory events")
    print(f"Found {len(fd_df)} FD count measurements")
    print(f"Found {len(resource_df)} resource count measurements")
    print(f"Found {len(pool_df)} pool stat measurements")
    print(f"Found {len(executor_df)} executor health measurements")
    if not executor_df.empty:
        print(f"  Agents with executors: {', '.join(executor_df['agent'].unique())}")
    
    print("Creating static plots...")
    fig = create_static_plots(trajectory_df, fd_df, resource_df, pool_df, executor_df)
    
    output_file = args.output or args.log_file.with_name(args.log_file.stem + '_static.html')
    print(f"Saving to {output_file}")
    fig.write_html(str(output_file))
    
    print(f"Static plots saved to {output_file}")
    print("Open it in a web browser to view the plots")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

