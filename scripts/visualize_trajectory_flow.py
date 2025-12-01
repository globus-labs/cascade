#!/usr/bin/env python3
"""
Interactive visualization of trajectory flow through agents and FD counts over time.

Parses log files to extract:
- Trajectory state transitions (which agent has which trajectory at what time)
- FD counts over time
- Agent relationships

Creates synchronized Plotly animations showing:
- Network graph with trajectories moving between agents
- FD count line plot over time
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Log patterns to extract trajectory state transitions
# These patterns track when trajectories move between agents
LOG_PATTERNS = {
    'dynamics_received': re.compile(
        r'Received advance spec for traj (\d+) chunk (\d+)'
    ),
    'dynamics_submit_audit': re.compile(
        r'Submitting audit for chunk (\d+) of traj (\d+)'
    ),
    'auditor_received': re.compile(
        r'Received chunk (\d+) from traj (\d+)',
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
    'sampler_received': re.compile(
        r'Sampling frames from chunk (\d+) of traj (\d+)'
    ),
    'trajectory_complete': re.compile(
        r'Traj (\d+) is complete'
    ),
    # Additional patterns for better tracking
    'submitted_to_dynamics': re.compile(
        r'submitted.*traj[ectory]*\s*(\d+)',
        re.IGNORECASE
    ),
    'submitting_to_audit': re.compile(
        r'submitting.*traj[ectory]*\s*(\d+).*chunk\s*(\d+).*audit',
        re.IGNORECASE
    ),
}

# FD count pattern
FD_PATTERN = re.compile(
    r'FD counts: total=(\d+), pipes=(\d+), files=(\d+), sockets=(\d+)'
)


def parse_log_file(log_file: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Parse log file to extract trajectory events and FD counts.
    
    Args:
        log_file: Path to log file
        
    Returns:
        Tuple of (trajectory_events_df, fd_counts_df)
    """
    trajectory_events = []
    fd_counts = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # Extract timestamp (assuming format like [2025-11-17 15:24:24.270])
            timestamp_match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\]', line)
            if not timestamp_match:
                continue
            
            timestamp_str = timestamp_match.group(1)
            try:
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
            except ValueError:
                continue
            
            # Extract agent name (assuming format like "INFO  (DynamicsEngine)")
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
                    elif event_type in ['auditor_received', 'audit_passed', 'audit_failed', 'sampler_received']:
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
            
            # Check for FD counts
            fd_match = FD_PATTERN.search(line)
            if fd_match:
                fd_counts.append({
                    'timestamp': timestamp,
                    'total': int(fd_match.group(1)),
                    'pipes': int(fd_match.group(2)),
                    'files': int(fd_match.group(3)),
                    'sockets': int(fd_match.group(4)),
                })
    
    trajectory_df = pd.DataFrame(trajectory_events)
    fd_df = pd.DataFrame(fd_counts)
    
    return trajectory_df, fd_df


def track_trajectory_states(trajectory_df: pd.DataFrame) -> Dict[int, List[Tuple[datetime, str]]]:
    """Track which agent has each trajectory at each point in time.
    
    Args:
        trajectory_df: DataFrame with trajectory events
        
    Returns:
        Dict mapping traj_id to list of (timestamp, agent) tuples
    """
    traj_states = defaultdict(list)
    
    for traj_id in trajectory_df['traj_id'].unique():
        traj_events = trajectory_df[trajectory_df['traj_id'] == traj_id].sort_values('timestamp')
        
        current_agent = None
        for _, event in traj_events.iterrows():
            event_type = event['event_type']
            timestamp = event['timestamp']
            
            # State transitions based on event type
            if event_type in ['dynamics_received', 'submitted_to_dynamics', 'next_chunk_submitted']:
                current_agent = 'DynamicsEngine'
            elif event_type in ['dynamics_submit_audit', 'submitting_to_audit']:
                # Transitioning to Auditor
                current_agent = 'Auditor'
            elif event_type == 'auditor_received':
                current_agent = 'Auditor'
            elif event_type == 'audit_failed':
                current_agent = 'DummySampler'
            elif event_type == 'audit_passed':
                # Check if there's a next chunk coming
                future_events = traj_events[traj_events['timestamp'] > timestamp]
                if any(future_events['event_type'] == 'next_chunk_submitted'):
                    # Will go back to DynamicsEngine
                    current_agent = 'DynamicsEngine'
                else:
                    # Trajectory complete, stays at Auditor
                    current_agent = 'Auditor'
            elif event_type == 'sampler_received':
                current_agent = 'DummySampler'
            elif event_type == 'trajectory_complete':
                # Mark as complete, remove from active tracking
                current_agent = None
            
            if current_agent:
                traj_states[traj_id].append((timestamp, current_agent))
    
    return dict(traj_states)


def get_trajectory_at_time(
    traj_states: Dict[int, List[Tuple[datetime, str]]],
    time: datetime
) -> Dict[str, List[int]]:
    """Get which trajectories are at each agent at a given time.
    
    Args:
        traj_states: Dict mapping traj_id to list of (timestamp, agent) tuples
        time: Time to query
        
    Returns:
        Dict mapping agent name to list of trajectory IDs at that agent
    """
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


def create_animation(
    trajectory_df: pd.DataFrame,
    fd_df: pd.DataFrame
) -> go.Figure:
    """Create synchronized animation of trajectory flow and FD counts.
    
    Args:
        trajectory_df: DataFrame with trajectory events
        fd_df: DataFrame with FD counts over time
        
    Returns:
        Plotly figure with synchronized animations
    """
    # Create subplots: text list on left, FD plot on right
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Trajectory Locations by Agent', 'File Descriptor Count Over Time'),
        column_widths=[0.5, 0.5],
        specs=[[{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # List of all agents we track
    all_agents = ['DynamicsEngine', 'Auditor', 'DummySampler', 'DummyLabeler', 'DummyTrainer', 'DatabaseMonitor']
    
    # Create initial text display (will be updated in frames)
    # We'll use annotations to display text, positioned vertically
    # Each agent gets a line showing its name and trajectories
    
    # Track trajectory states over time
    traj_states = {}
    if not trajectory_df.empty:
        traj_states = track_trajectory_states(trajectory_df)
    
    # Add initial empty trace for text subplot (must be first to maintain trace order)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, len(all_agents)],
            mode='markers',
            marker=dict(size=0, opacity=0),
            showlegend=False,
        ),
        row=1, col=1
    )
    
    # Add FD count plot (static, will be animated)
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
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=fd_df['timestamp'],
                y=fd_df['pipes'],
                mode='lines',
                name='Pipes',
                line=dict(color='blue', width=1),
                showlegend=True,
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=fd_df['timestamp'],
                y=fd_df['files'],
                mode='lines',
                name='Files',
                line=dict(color='green', width=1),
                showlegend=True,
            ),
            row=1, col=2
        )
    else:
        # Add empty FD traces if no data
        fig.add_trace(
            go.Scatter(x=[], y=[], mode='lines+markers', name='Total FDs', line=dict(color='red', width=2), showlegend=True),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=[], y=[], mode='lines', name='Pipes', line=dict(color='blue', width=1), showlegend=True),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=[], y=[], mode='lines', name='Files', line=dict(color='green', width=1), showlegend=True),
            row=1, col=2
        )
    
    # Update layout for text display
    fig.update_layout(
        title='Trajectory Flow and FD Count Animation',
        height=600,
        xaxis_title='',
        yaxis_title='',
        xaxis2_title='Time',
        yaxis2_title='FD Count',
        # Hide axes for text display
        xaxis=dict(
            range=[0, 1],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        yaxis=dict(
            range=[0, len(all_agents)],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
    )
    
    # Create animation frames
    if traj_states:
        # Determine time range
        if not fd_df.empty:
            min_time = min(trajectory_df['timestamp'].min(), fd_df['timestamp'].min())
            max_time = max(trajectory_df['timestamp'].max(), fd_df['timestamp'].max())
        else:
            min_time = trajectory_df['timestamp'].min()
            max_time = trajectory_df['timestamp'].max()
        
        # Create time steps (every 2 seconds)
        time_steps = pd.date_range(start=min_time, end=max_time, freq='2S')
        
        frames = []
        for t in time_steps:
            # Get trajectories at each agent at this time
            trajs_by_agent = get_trajectory_at_time(traj_states, t)
            
            # Get FD data up to this time
            fd_up_to_time = fd_df[fd_df['timestamp'] <= t] if not fd_df.empty else pd.DataFrame()
            
            # Build frame data
            frame_data = []
            
            # Create text annotations for each agent
            annotations = []
            for i, agent_name in enumerate(all_agents):
                traj_list = trajs_by_agent.get(agent_name, [])
                if traj_list:
                    traj_str = ', '.join(map(str, sorted(traj_list)))
                    text = f"<b>{agent_name}:</b> {traj_str}"
                else:
                    text = f"<b>{agent_name}:</b> (empty)"
                
                annotations.append(
                    dict(
                        x=0.05,
                        y=len(all_agents) - i - 0.5,
                        xref='x',
                        yref='y',
                        text=text,
                        showarrow=False,
                        font=dict(size=14, color='black'),
                        align='left',
                        xanchor='left',
                    )
                )
            
            # Frame data must match initial trace order:
            # Trace 0: Text scatter (row=1, col=1)
            # Trace 1: FD total (row=1, col=2)
            # Trace 2: FD pipes (row=1, col=2)
            # Trace 3: FD files (row=1, col=2)
            
            # Trace 0: Text scatter (empty, annotations handle the display)
            frame_data.append(
                go.Scatter(
                    x=[0, 1],
                    y=[0, len(all_agents)],
                    mode='markers',
                    marker=dict(size=0, opacity=0),
                    showlegend=False,
                )
            )
            
            # Traces 1-3: FD plots
            if not fd_up_to_time.empty:
                frame_data.append(
                    go.Scatter(
                        x=fd_up_to_time['timestamp'],
                        y=fd_up_to_time['total'],
                        mode='lines+markers',
                        name='Total FDs',
                        line=dict(color='red', width=2),
                        showlegend=True,
                    )
                )
                frame_data.append(
                    go.Scatter(
                        x=fd_up_to_time['timestamp'],
                        y=fd_up_to_time['pipes'],
                        mode='lines',
                        name='Pipes',
                        line=dict(color='blue', width=1),
                        showlegend=True,
                    )
                )
                frame_data.append(
                    go.Scatter(
                        x=fd_up_to_time['timestamp'],
                        y=fd_up_to_time['files'],
                        mode='lines',
                        name='Files',
                        line=dict(color='green', width=1),
                        showlegend=True,
                    )
                )
            else:
                # Add empty traces to maintain trace count
                frame_data.append(go.Scatter(x=[], y=[], mode='lines+markers', name='Total FDs', line=dict(color='red', width=2), showlegend=True))
                frame_data.append(go.Scatter(x=[], y=[], mode='lines', name='Pipes', line=dict(color='blue', width=1), showlegend=True))
                frame_data.append(go.Scatter(x=[], y=[], mode='lines', name='Files', line=dict(color='green', width=1), showlegend=True))
            
            # Create frame with annotations
            frame = go.Frame(data=frame_data, name=str(t), layout=go.Layout(annotations=annotations))
            frames.append(frame)
        
        # Add initial annotations
        initial_annotations = []
        for i, agent_name in enumerate(all_agents):
            initial_annotations.append(
                dict(
                    x=0.05,
                    y=len(all_agents) - i - 0.5,
                    xref='x',
                    yref='y',
                    text=f"<b>{agent_name}:</b> (empty)",
                    showarrow=False,
                    font=dict(size=14, color='black'),
                    align='left',
                    xanchor='left',
                )
            )
        
        fig.frames = frames
        
        # Add animation controls
        fig.update_layout(
            annotations=initial_annotations,
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 200, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 100}
                        }]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ],
                'x': 0.1,
                'y': 0,
                'xanchor': 'left',
                'yanchor': 'bottom'
            }],
            sliders=[{
                'active': 0,
                'yanchor': 'top',
                'xanchor': 'left',
                'currentvalue': {
                    'prefix': 'Time: ',
                    'visible': True,
                    'xanchor': 'right'
                },
                'transition': {'duration': 100},
                'steps': [
                    {
                        'args': [[f.name], {
                            'frame': {'duration': 200, 'redraw': True},
                            'transition': {'duration': 100}
                        }],
                        'label': f.name,
                        'method': 'animate'
                    }
                    for f in frames
                ]
            }]
        )
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Visualize trajectory flow through agents and FD counts over time'
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
        help='Output HTML file (default: log_file with .html extension)'
    )
    
    args = parser.parse_args()
    
    if not args.log_file.exists():
        print(f"Error: Log file {args.log_file} does not exist")
        return 1
    
    print(f"Parsing log file: {args.log_file}")
    trajectory_df, fd_df = parse_log_file(args.log_file)
    
    print(f"Found {len(trajectory_df)} trajectory events")
    print(f"Found {len(fd_df)} FD count measurements")
    
    print("Creating visualization...")
    fig = create_animation(trajectory_df, fd_df)
    
    output_file = args.output or args.log_file.with_suffix('.html')
    print(f"Saving to {output_file}")
    fig.write_html(str(output_file))
    
    print(f"Visualization saved to {output_file}")
    print("Open it in a web browser to view the interactive animation")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

