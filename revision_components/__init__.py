"""
Revision Components

Modular components for different types of website revision:
- CUA failure revision (from failed trajectories)
- Unsupported tasks revision (from Stage 1 judge)
- Integrated two-phase revision
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List

class RevisionComponent(ABC):
    """Base class for revision components"""
    
    def __init__(self, coder, commenter=None):
        self.coder = coder
        self.commenter = commenter
    
    @abstractmethod
    async def revise(self, model_name: str, app_name: str, v0_html: str, 
                    context: Dict[str, Any], mcts: bool = False, 
                    destylized: bool = False, v0_dir: str = None) -> Dict[str, Any]:
        """Execute revision based on component-specific logic
        
        Args:
            model_name: Model to use for revision
            app_name: Application name
            v0_html: Original initial HTML content
            context: Component-specific context data
            mcts: Whether to generate 3 versions and select best (only for CUA)
            destylized: Whether to apply destylization (only for CUA)
            v0_dir: Initial data directory name (stored under initial/[dir])
            
        Returns:
            Dict with success, html_content, and metadata
        """
        pass
