"""
Create animated GIFs showing code evolution through checkpoints with syntax-highlighted diffs.
"""
import re
from pathlib import Path
from typing import List, Tuple, Dict, Literal
import difflib
from PIL import Image, ImageDraw, ImageFont
from pygments import lex
from pygments.lexers import PythonLexer
from pygments.token import Token


def get_vscode_dark_colors() -> Dict:
    """VS Code Dark+ theme colors for syntax highlighting."""
    return {
        Token.Keyword: (197, 134, 192),           # Purple for keywords
        Token.Keyword.Namespace: (197, 134, 192), # Purple for import
        Token.Name.Function: (220, 220, 170),     # Yellow for function names
        Token.Name.Class: (78, 201, 176),         # Teal for class names
        Token.Name.Builtin: (220, 220, 170),      # Yellow for built-in functions
        Token.String: (206, 145, 120),            # Orange/brown for strings
        Token.Comment: (106, 153, 85),            # Green for comments
        Token.Number: (181, 206, 168),            # Light green for numbers
        Token.Operator: (212, 212, 212),          # White for operators
        Token.Name: (156, 220, 254),              # Light blue for variable names
        Token.Punctuation: (212, 212, 212),       # White for punctuation
        Token.Text: (212, 212, 212),              # Default white
        Token.Literal: (181, 206, 168),           # Light green for literals
    }


def get_vscode_light_colors() -> Dict:
    """VS Code Light+ theme colors for syntax highlighting."""
    return {
        Token.Keyword: (175, 0, 219),             # Purple for keywords
        Token.Keyword.Namespace: (175, 0, 219),   # Purple for import
        Token.Name.Function: (121, 94, 38),       # Brown for function names
        Token.Name.Class: (38, 127, 153),         # Teal for class names
        Token.Name.Builtin: (121, 94, 38),        # Brown for built-in functions
        Token.String: (163, 21, 21),              # Red for strings
        Token.Comment: (0, 128, 0),               # Green for comments
        Token.Number: (9, 134, 88),               # Green for numbers
        Token.Operator: (0, 0, 0),                # Black for operators
        Token.Name: (0, 0, 255),                  # Blue for variable names
        Token.Punctuation: (0, 0, 0),             # Black for punctuation
        Token.Text: (0, 0, 0),                    # Default black
        Token.Literal: (9, 134, 88),              # Green for literals
    }


def extract_checkpoint_number(checkpoint_dir: str) -> int:
    """Extract the checkpoint number from a directory name."""
    match = re.search(r'checkpoint_(\d+)', checkpoint_dir)
    return int(match.group(1)) if match else 0


def get_checkpoint_dirs(run_dir: Path) -> List[Path]:
    """Get all checkpoint directories sorted by checkpoint number."""
    checkpoints_dir = run_dir / "evolution_results" / "checkpoints"
    if not checkpoints_dir.exists():
        return []
    
    checkpoint_dirs = [
        d for d in checkpoints_dir.iterdir() 
        if d.is_dir() and d.name.startswith("checkpoint_")
    ]
    
    # Sort by checkpoint number
    checkpoint_dirs.sort(key=lambda x: extract_checkpoint_number(x.name))
    return checkpoint_dirs


def compute_line_diffs(old_lines: List[str], new_lines: List[str]) -> List[Tuple[str, str]]:
    """
    Compute line-by-line diffs.
    Returns a list of (status, line) where status is 'add', 'remove', or 'same'.
    """
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
    result = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for line in new_lines[j1:j2]:
                result.append(('same', line))
        elif tag == 'delete':
            for line in old_lines[i1:i2]:
                result.append(('remove', line))
        elif tag == 'insert':
            for line in new_lines[j1:j2]:
                result.append(('add', line))
        elif tag == 'replace':
            # Show deletions first, then additions
            for line in old_lines[i1:i2]:
                result.append(('remove', line))
            for line in new_lines[j1:j2]:
                result.append(('add', line))
    
    return result


def tokenize_line(line: str, lexer) -> List[Tuple[Token, str]]:
    """Tokenize a line of code using pygments."""
    return list(lex(line, lexer))


def draw_syntax_highlighted_line(
    draw: ImageDraw.Draw,
    line: str,
    x: int,
    y: int,
    font: ImageFont.FreeTypeFont,
    status: str,
    lexer,
    syntax_colors: Dict
):
    """Draw a line of code with syntax highlighting."""
    # Base colors for diff status (used for tinting)
    status_tints = {
        'same': 1.0,      # No tint
        'add': 1.1,       # Slightly brighter
        'remove': 1.0,    # Normal
    }
    
    tint = status_tints[status]
    tokens = tokenize_line(line, lexer)
    
    current_x = x
    for token_type, token_value in tokens:
        # Find the most specific color match
        color = syntax_colors.get(token_type)
        if color is None:
            # Try parent token types
            for parent_type in [Token.Text, Token]:
                if token_type in parent_type:
                    color = syntax_colors.get(parent_type, (212, 212, 212))
                    break
            else:
                color = (212, 212, 212)  # Default
        
        # Apply tint for additions
        if status == 'add':
            color = tuple(min(255, int(c * tint)) for c in color)
        elif status == 'remove':
            # For removals, tint towards red
            color = tuple(min(255, int(c * 0.8 + 255 * 0.2)) for c in color)
        
        # Draw the token
        draw.text((current_x, y), token_value, fill=color, font=font)
        
        # Calculate width for next token
        bbox = draw.textbbox((current_x, y), token_value, font=font)
        current_x = bbox[2]


def create_code_image(
    lines_with_status: List[Tuple[str, str]],
    title: str,
    width: int = 1400,
    canvas_height: int = None,
    line_height: int = 30,
    font_size: int = 16,
    theme: Literal['dark', 'light'] = 'dark'
) -> Image.Image:
    """
    Create an image showing code with diff highlighting and syntax coloring.
    
    Args:
        lines_with_status: List of (status, line) tuples
        title: Title to show at the top
        width: Image width in pixels
        canvas_height: Total canvas height override (if None, auto-calculated)
        line_height: Height of each line in pixels
        font_size: Font size for the code
        theme: Color theme - 'dark' (VS Code Dark+) or 'light' (VS Code Light+)
    """
    # Calculate image height - ensure ALL lines fit
    padding = 60
    title_height = 100
    bottom_padding = 950  # Large buffer to fit all diff lines
    
    if canvas_height is None:
        height = title_height + len(lines_with_status) * line_height + bottom_padding
    else:
        height = canvas_height
    
    # Theme-specific settings
    if theme == 'dark':
        bg_color = (30, 30, 30)
        title_color = (255, 255, 255)
        syntax_colors = get_vscode_dark_colors()
        bg_colors = {
            'same': None,
            'add': (0, 70, 0),
            'remove': (70, 0, 0),
        }
    else:  # light
        bg_color = (255, 255, 255)
        title_color = (0, 0, 0)
        syntax_colors = get_vscode_light_colors()
        bg_colors = {
            'same': None,
            'add': (200, 255, 200),
            'remove': (255, 200, 200),
        }
    
    # Create image
    img = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Try to use a monospace font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Monaco.ttf", font_size)
        title_font = ImageFont.truetype("/System/Library/Fonts/Monaco.ttf", 24)
    except:
        try:
            font = ImageFont.truetype("/Library/Fonts/Courier New.ttf", font_size)
            title_font = ImageFont.truetype("/Library/Fonts/Courier New.ttf", 24)
        except:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()
    
    # Draw title
    draw.text((padding, 25), title, fill=title_color, font=title_font)
    
    # Get syntax colors and lexer
    lexer = PythonLexer()
    
    # Draw each line
    y_offset = title_height
    for idx, (status, line) in enumerate(lines_with_status):
        # Expand tabs
        line = line.replace('\t', '    ')
        
        # Draw background for changed lines
        if bg_colors[status]:
            draw.rectangle(
                [(0, y_offset - 2), (width, y_offset + line_height - 2)],
                fill=bg_colors[status]
            )
        
        # Draw the line with syntax highlighting
        draw_syntax_highlighted_line(
            draw, line, padding, y_offset, font, status, lexer, syntax_colors
        )
        
        y_offset += line_height
    
    return img


def create_diff_gif(
    run_dir: Path,
    output_path: Path = None,
    initial_program_path: Path = None,
    duration_per_frame: int = 2000,
    width: int = 1400,
    canvas_height: int = None,
    line_height: int = 30,
    font_size: int = 16,
    theme: Literal['dark', 'light'] = 'dark',
    include_final_best: bool = True
):
    """
    Create a GIF showing the evolution of code through checkpoints.
    
    Args:
        run_dir: Path to the run directory (e.g., 'opt/')
        output_path: Path where the GIF should be saved (default: run_dir/evolution_results/code_evolution_diff.gif)
        initial_program_path: Path to initial_program.py (default: run_dir/initial_program.py)
        duration_per_frame: Duration of each frame in milliseconds
        width: Image width in pixels
        canvas_height: Total canvas height override (if None, auto-calculated per frame)
        line_height: Height of each line in pixels
        font_size: Font size for the code
        theme: Color theme - 'dark' (default, VS Code Dark+) or 'light' (VS Code Light+)
        include_final_best: If True, adds a final frame with the best program (no diffs)
    
    Returns:
        Path to the created GIF file
    """
    run_dir = Path(run_dir)
    
    # Set default paths
    if initial_program_path is None:
        initial_program_path = run_dir / "initial_program.py"
    
    if output_path is None:
        output_path = run_dir / "evolution_results" / "code_evolution_diff.gif"
    
    if not initial_program_path.exists():
        raise FileNotFoundError(f"Initial program not found at {initial_program_path}")
    
    # Get checkpoint directories
    checkpoint_dirs = get_checkpoint_dirs(run_dir)
    if not checkpoint_dirs:
        raise ValueError(f"No checkpoint directories found in {run_dir}")
    
    print(f"Found {len(checkpoint_dirs)} checkpoints")
    
    # Read initial program
    with open(initial_program_path, 'r') as f:
        current_code = f.read()
    
    images = []
    
    # Create first frame with initial program
    current_lines = current_code.splitlines()
    lines_with_status = [('same', line) for line in current_lines]
    img = create_code_image(
        lines_with_status,
        "Initial Program",
        width=width,
        canvas_height=canvas_height,
        line_height=line_height,
        font_size=font_size,
        theme=theme
    )
    images.append(img)
    print(f"Created frame for initial program")
    
    # Create frame for each checkpoint
    for checkpoint_dir in checkpoint_dirs:
        best_program_path = checkpoint_dir / "best_program.py"
        
        if not best_program_path.exists():
            print(f"Warning: {best_program_path} not found, skipping")
            continue
        
        # Read new code
        with open(best_program_path, 'r') as f:
            new_code = f.read()
        
        new_lines = new_code.splitlines()
        
        # Compute diffs
        lines_with_status = compute_line_diffs(current_lines, new_lines)
        
        # Create image
        checkpoint_name = checkpoint_dir.name
        title = f"{checkpoint_name.replace('_', ' ').title()}"
        img = create_code_image(
            lines_with_status,
            title,
            width=width,
            canvas_height=canvas_height,
            line_height=line_height,
            font_size=font_size,
            theme=theme
        )
        images.append(img)
        print(f"Created frame for {checkpoint_name}")
        
        # Update current code for next iteration
        current_lines = new_lines
        current_code = new_code
    
    # Add final frame with the best program (no diffs)
    if include_final_best:
        best_dir = run_dir / "evolution_results" / "best"
        if best_dir.exists():
            final_best_path = best_dir / "best_program.py"
            if final_best_path.exists():
                with open(final_best_path, 'r') as f:
                    final_code = f.read()
                
                final_lines = final_code.splitlines()
                # Mark all lines as 'same' - no diff highlighting
                lines_with_status = [('same', line) for line in final_lines]
                
                img = create_code_image(
                    lines_with_status,
                    "Final Best Program",
                    width=width,
                    canvas_height=canvas_height,
                    line_height=line_height,
                    font_size=font_size,
                    theme=theme
                )
                images.append(img)
                print(f"Created frame for final best program")
            else:
                print(f"Warning: {final_best_path} not found")
    
    # Save as GIF
    if images:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration_per_frame,
            loop=0,
            optimize=False
        )
        print(f"\nGIF saved to {output_path}")
        print(f"Total frames: {len(images)}")
        print(f"Duration per frame: {duration_per_frame}ms")
        print(f"Total duration: {len(images) * duration_per_frame / 1000:.1f}s")
        return output_path
    else:
        raise RuntimeError("No images created!")
