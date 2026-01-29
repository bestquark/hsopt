"""Utility functions for parsing OpenEvolve log files and checkpoints."""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np


def openevolve_log_parser(log_path: str) -> Tuple[List[int], List[float], Dict[str, Dict[str, List[float]]]]:
    """
    Parse OpenEvolve log file to extract iteration data and constraint details.
    
    Parameters
    ----------
    log_path : str
        Path to the OpenEvolve log file
        
    Returns
    -------
    iterations : List[int]
        List of iteration numbers
    scores : List[float]
        List of combined scores for each iteration
    details : Dict[str, Dict[str, List[float]]]
        Dictionary mapping constraint names to their values and scores
        Format: {constraint_name: {'values': [...], 'scores': [...]}}
        
    Examples
    --------
    >>> iterations, scores, details = openevolve_log_parser('logs/openevolve_20260127.log')
    >>> print(f"Iterations: {iterations}")
    >>> print(f"Final score: {scores[-1]}")
    >>> print(f"Time constraint values: {details['time']['values']}")
    """
    log_path = Path(log_path)
    
    with open(log_path, 'r') as f:
        log_content = f.read()
    
    # Initialize data structures
    iterations = []
    scores = []
    details = {}
    
    # Regex patterns
    metrics_pattern = r"Metrics: combined_score=([\d.]+), std_score=([\d.]+), all_satisfied=([\d.]+), details=(\{.+\})"
    iteration_pattern = r'Iteration (\d+):'
    
    # Parse log file line by line
    lines = log_content.split('\n')
    for i, line in enumerate(lines):
        iter_match = re.search(iteration_pattern, line)
        if iter_match and i + 1 < len(lines):
            next_line = lines[i + 1]
            metrics_match = re.search(metrics_pattern, next_line)
            if metrics_match:
                iteration = int(iter_match.group(1))
                combined_score = float(metrics_match.group(1))
                details_str = metrics_match.group(4)
                
                # Parse the details dictionary
                # Replace Python boolean/quote syntax with JSON syntax
                details_str = details_str.replace("'", '"').replace('True', 'true').replace('False', 'false')
                try:
                    details_dict = json.loads(details_str)
                    
                    iterations.append(iteration)
                    scores.append(combined_score)
                    
                    # Extract constraint values and scores
                    for constraint_name, constraint_data in details_dict.items():
                        if constraint_name not in details:
                            details[constraint_name] = {'values': [], 'scores': []}
                        details[constraint_name]['values'].append(constraint_data['value'])
                        details[constraint_name]['scores'].append(constraint_data['score'])
                except json.JSONDecodeError as e:
                    # Skip lines that can't be parsed
                    pass
    
    return iterations, scores, details


def get_latest_openevolve_log(log_dir: str = "opt/evolution_results/logs") -> Path:
    """
    Find the most recent OpenEvolve log file in a directory.
    
    Parameters
    ----------
    log_dir : str
        Directory containing OpenEvolve log files
        
    Returns
    -------
    Path
        Path to the most recent log file
        
    Raises
    ------
    FileNotFoundError
        If no log files are found in the directory
        
    Examples
    --------
    >>> log_path = get_latest_openevolve_log()
    >>> print(f"Latest log: {log_path}")
    """
    log_dir = Path(log_dir)
    log_files = sorted(log_dir.glob("openevolve_*.log"))
    
    if not log_files:
        raise FileNotFoundError(f"No OpenEvolve log files found in {log_dir}")
    
    return log_files[-1]


def parse_checkpoint(checkpoint_dir: str) -> Dict:
    """
    Parse a single OpenEvolve checkpoint directory.
    
    Parameters
    ----------
    checkpoint_dir : str
        Path to checkpoint directory (e.g., "checkpoints/checkpoint_10")
        
    Returns
    -------
    Dict
        Dictionary with checkpoint data including:
        - iteration: checkpoint iteration number
        - best_program_id: UUID of best program
        - best_metrics: metrics dict for best program
        - metadata: full metadata including MAP-Elites grid
        - num_programs: total number of programs evaluated
        
    Examples
    --------
    >>> data = parse_checkpoint("opt/evolution_results/checkpoints/checkpoint_10")
    >>> print(f"Best score at iteration {data['iteration']}: {data['best_metrics']['combined_score']}")
    """
    checkpoint_path = Path(checkpoint_dir)
    
    # Read metadata
    metadata_path = checkpoint_path / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Read best program info
    best_info_path = checkpoint_path / "best_program_info.json"
    with open(best_info_path, 'r') as f:
        best_info = json.load(f)
    
    # Extract iteration number from directory name
    iteration = int(checkpoint_path.name.split('_')[-1])
    
    return {
        'iteration': iteration,
        'best_program_id': metadata['best_program_id'],
        'best_metrics': best_info,
        'metadata': metadata,
        'num_programs': len(metadata['archive']),
        'island_feature_maps': metadata['island_feature_maps'],
        'feature_stats': metadata.get('feature_stats', {}),
    }


def parse_all_checkpoints(checkpoints_dir: str = "opt/evolution_results/checkpoints") -> List[Dict]:
    """
    Parse all checkpoints in a directory and return sorted by iteration.
    
    Parameters
    ----------
    checkpoints_dir : str
        Directory containing checkpoint subdirectories
        
    Returns
    -------
    List[Dict]
        List of checkpoint data dicts sorted by iteration
        
    Examples
    --------
    >>> checkpoints = parse_all_checkpoints()
    >>> for cp in checkpoints:
    ...     print(f"Iteration {cp['iteration']}: score={cp['best_metrics']['combined_score']:.4f}")
    """
    checkpoints_path = Path(checkpoints_dir)
    checkpoint_dirs = sorted(checkpoints_path.glob("checkpoint_*"))
    
    checkpoints = []
    for cp_dir in checkpoint_dirs:
        try:
            data = parse_checkpoint(cp_dir)
            checkpoints.append(data)
        except Exception as e:
            print(f"Warning: Failed to parse {cp_dir}: {e}")
    
    # Sort by iteration
    checkpoints.sort(key=lambda x: x['iteration'])
    return checkpoints


def extract_evolution_metrics(checkpoints: List[Dict]) -> Dict[str, List]:
    """
    Extract time-series metrics from parsed checkpoints.
    
    Parameters
    ----------
    checkpoints : List[Dict]
        List of checkpoint data from parse_all_checkpoints()
        
    Returns
    -------
    Dict[str, List]
        Dictionary with lists of metrics over iterations:
        - iterations: iteration numbers
        - combined_scores: best combined score at each checkpoint
        - times: execution time values
        - errors: error values
        - scores_by_constraint: dict mapping constraint name to score list
        
    Examples
    --------
    >>> checkpoints = parse_all_checkpoints()
    >>> metrics = extract_evolution_metrics(checkpoints)
    >>> plt.plot(metrics['iterations'], metrics['combined_scores'])
    """
    iterations = []
    combined_scores = []
    times = []
    errors = []
    scores_by_constraint = {}
    
    for cp in checkpoints:
        iterations.append(cp['iteration'])
        best_info = cp['best_metrics']
        
        # Metrics are nested under 'metrics' key in best_program_info.json
        metrics_data = best_info.get('metrics', best_info)
        
        combined_scores.append(metrics_data.get('combined_score', 0.0))
        
        # Extract constraint details
        details = metrics_data.get('details', {})
        for constraint_name, constraint_data in details.items():
            if constraint_name not in scores_by_constraint:
                scores_by_constraint[constraint_name] = {'values': [], 'scores': []}
            scores_by_constraint[constraint_name]['values'].append(constraint_data.get('value', 0.0))
            scores_by_constraint[constraint_name]['scores'].append(constraint_data.get('score', 0.0))
        
        # Convenience extraction
        if 'time' in details:
            times.append(details['time']['value'])
        if 'error' in details:
            errors.append(details['error']['value'])
    
    return {
        'iterations': iterations,
        'combined_scores': combined_scores,
        'times': times,
        'errors': errors,
        'scores_by_constraint': scores_by_constraint,
    }


def extract_map_elites_grid(checkpoint: Dict, checkpoint_dir: str = None, grid_size: int = 10) -> np.ndarray:
    """
    Extract MAP-Elites grid from a checkpoint for visualization.
    
    Parameters
    ----------
    checkpoint : Dict
        Checkpoint data from parse_checkpoint()
    checkpoint_dir : str, optional
        Path to checkpoint directory to load program scores
    grid_size : int
        Size of the MAP-Elites grid (default: 10x10)
        
    Returns
    -------
    np.ndarray
        2D array (grid_size x grid_size) with combined scores
        NaN for unoccupied cells
        
    Examples
    --------
    >>> cp = parse_checkpoint("opt/evolution_results/checkpoints/checkpoint_100")
    >>> grid = extract_map_elites_grid(cp, "opt/evolution_results/checkpoints/checkpoint_100")
    >>> plt.imshow(grid, origin='lower', cmap='viridis')
    """
    grid = np.full((grid_size, grid_size), np.nan)
    
    # Get island 0's feature map (typically the main island)
    island_maps = checkpoint['island_feature_maps']
    if not island_maps or len(island_maps) == 0:
        return grid
    
    feature_map = island_maps[0]  # Island 0
    
    # Try to load actual scores from program JSON files
    if checkpoint_dir:
        programs_dir = Path(checkpoint_dir) / "programs"
        
        for cell_key, program_id in feature_map.items():
            # Parse cell coordinates
            parts = cell_key.split('-')
            if len(parts) == 2:
                try:
                    x, y = int(parts[0]), int(parts[1])
                    if 0 <= x < grid_size and 0 <= y < grid_size:
                        # Try to load program JSON to get score
                        program_json = programs_dir / f"{program_id}.json"
                        if program_json.exists():
                            with open(program_json, 'r') as f:
                                program_data = json.load(f)
                                score = program_data.get('metrics', {}).get('combined_score', 0.0)
                                grid[y, x] = score
                        else:
                            grid[y, x] = 0.5  # Fallback if JSON not found
                except (ValueError, json.JSONDecodeError, KeyError):
                    continue
    else:
        # Fallback: just mark cells as occupied
        for cell_key in feature_map.keys():
            parts = cell_key.split('-')
            if len(parts) == 2:
                try:
                    x, y = int(parts[0]), int(parts[1])
                    if 0 <= x < grid_size and 0 <= y < grid_size:
                        grid[y, x] = 1.0
                except ValueError:
                    continue
    
    return grid
