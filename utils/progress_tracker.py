import asyncio
import os
import time
import threading
import shutil
from typing import Dict, List
from datetime import datetime

class ProgressTracker:
    def __init__(self, stage_name: str, models: List[str], apps: List[str]):
        """Terminal progress tracker optimized for parallel task display."""
        self.stage_name = stage_name
        self.models = models
        self.apps = apps
        self.status_matrix = {}
        self.error_details = {}  # Store detailed error information
        self.retry_details = {}  # Store retry details
        self.timing_info = {}  # Store timing information
        self.analysis_info = {}  # Store analysis entries
        self.start_time = time.time()
        self.running = True
        self.lock = threading.Lock()  # Thread-safe lock
        self.last_update = time.time()
        # Grid paging & sizing
        self.page_index = 0
        self.page_interval = float(os.environ.get("BUI_PAGE_INTERVAL", "3.0"))
        self.last_page_switch = time.time()
        self._small_screen = False
        self._term_cols = shutil.get_terminal_size((100, 40)).columns
        self._term_lines = shutil.get_terminal_size((100, 40)).lines
        
        # Initialize status matrix
        for model in models:
            self.status_matrix[model] = {}
            self.error_details[model] = {}
            self.retry_details[model] = {}
            self.timing_info[model] = {}
            self.analysis_info[model] = {}
            for app in apps:
                self.status_matrix[model][app] = "⏳ Waiting"
                self.error_details[model][app] = None
                self.retry_details[model][app] = None
                self.timing_info[model][app] = []
                self.analysis_info[model][app] = []
    
    def update_status(self, model: str, app: str, status: str, error_detail: str = None, retry_info: dict = None):
        """Thread-safe update of a single model/app cell."""
        with self.lock:
            if model in self.status_matrix and app in self.status_matrix[model]:
                self.status_matrix[model][app] = status
                if error_detail:
                    self.error_details[model][app] = error_detail
                    # Persist into timing log so it is not lost on refresh
                    ts = datetime.now().strftime("%H:%M:%S")
                    short_err = error_detail if len(error_detail) <= 150 else (error_detail[:147] + "...")
                    self.timing_info[model][app].append(f"[{ts}] ERROR: {short_err}")
                if retry_info:
                    self.retry_details[model][app] = retry_info
                    # Write retry summary into timing log to avoid losing details
                    ts = datetime.now().strftime("%H:%M:%S")
                    if isinstance(retry_info, list):
                        # Record individual attempts for clearer visualization
                        for at in retry_info:
                            attempt = at.get('attempt', '?')
                            success = at.get('success', False)
                            gen_time = at.get('generation_time', None)
                            html_len = at.get('html_length', None)
                            status_icon = "✅" if success else "❌"
                            parts = [f"Attempt {attempt}"]
                            if gen_time is not None:
                                parts.append(f"{gen_time}s")
                            if html_len is not None:
                                parts.append(f"{html_len} chars")
                            summary = " ".join(parts)
                            self.timing_info[model][app].append(f"[{ts}] RETRY {status_icon}: {summary}")
                    else:
                        summary = str(retry_info)
                        short_summary = summary if len(summary) <= 150 else (summary[:147] + "...")
                        self.timing_info[model][app].append(f"[{ts}] RETRY: {short_summary}")
                self.last_update = time.time()
    
    def add_timing_info(self, model: str, app: str, timing_text: str):
        """Append timing information for a given model/app."""
        with self.lock:
            if model in self.timing_info and app in self.timing_info[model]:
                from datetime import datetime
                ts = datetime.now().strftime("%H:%M:%S")
                self.timing_info[model][app].append(f"[{ts}] {timing_text}")
                self.last_update = time.time()
    
    def add_analysis_info(self, model: str, app: str, analysis_text: str):
        """Append analysis content for a given model/app."""
        with self.lock:
            if model in self.analysis_info and app in self.analysis_info[model]:
                # Truncate long analysis to 200 chars for display
                truncated = analysis_text[:200] + "..." if len(analysis_text) > 200 else analysis_text
                from datetime import datetime
                ts = datetime.now().strftime("%H:%M:%S")
                self.analysis_info[model][app].append(f"[{ts}] {truncated}")
                self.last_update = time.time()
    
    def stop(self):
        """Stop the display loop."""
        self.running = False
    
    async def display_loop(self):
        """Display loop that periodically refreshes terminal output."""
        last_display_time = 0
        while self.running:
            # Poll frequently but refresh only when state changes
            current_time = time.time()
            if (current_time - last_display_time >= 1.0) or (self.last_update > last_display_time):
                self._display_matrix()
                last_display_time = current_time
            await asyncio.sleep(0.5)  # Check every 0.5 seconds
        
        # Final paint once after stop
        self._display_matrix()
    
    def _display_matrix(self):
        """Render the status matrix and associated diagnostics."""
        with self.lock:  # Ensure thread-safe reads while rendering
            # Clear screen and move cursor to top
            os.system('clear' if os.name == 'posix' else 'cls')
            # Read terminal size
            size = shutil.get_terminal_size((100, 40))
            self._term_cols, self._term_lines = size.columns, size.lines
            self._small_screen = self._term_cols < 100 or self._term_lines < 40
            # Rotate pages when multiple pages exist
            now = time.time()
            if now - self.last_page_switch >= self.page_interval:
                self.page_index += 1
                self.last_page_switch = now
            
            elapsed = time.time() - self.start_time
            elapsed_str = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"
            current_time = datetime.now().strftime("%H:%M:%S")
            
            print(f"🚀 {self.stage_name}")
            print(f"⏰ Time: {current_time} | Elapsed: {elapsed_str}")
            print("=" * self._term_cols)
            
            # Dynamically compute column width and paging
            model_col_width = 12
            available_width = max(20, self._term_cols - model_col_width - 2)
            base_col_width = 18  # Moderate column width to avoid a cramped grid
            col_width = max(12, min(28, base_col_width))
            # Compute how many app columns can fit per page
            apps_per_page = max(1, available_width // col_width)
            # Exclude special BATCH column from visible grid
            display_apps_all = [a for a in self.apps if a != "BATCH"]
            total_apps = len(display_apps_all)
            total_pages = max(1, (total_apps + apps_per_page - 1) // apps_per_page) if total_apps else 1
            if total_pages == 1:
                self.page_index = 0
            page = self.page_index % total_pages
            start_idx = page * apps_per_page
            end_idx = min(total_apps, start_idx + apps_per_page)
            visible_apps = display_apps_all[start_idx:end_idx]
            
            # Header row
            header = f"{'Model':<{model_col_width}}"
            for app in visible_apps:
                app_display = app if len(app) <= col_width - 2 else app[:max(1, col_width-5)] + "..."
                header += f"{app_display:<{col_width}}"
            print(header)
            # Page info
            if total_apps > 0 and total_pages > 1:
                page_info = f"Apps {start_idx+1}-{end_idx} of {total_apps} (page {page+1}/{total_pages})"
                print(page_info)
            print("-" * self._term_cols)
            
            # Status rows
            for model in self.models:
                row = f"{model:<{model_col_width}}"
                for app in visible_apps:
                    status = self.status_matrix[model][app]
                    # Aggressively truncate to fit small grids; full errors go to detail sections
                    if len(status) > col_width - 1:
                        if "✏️" in status and "Generating" in status:
                            status = "✏️ Gen..."
                        elif "💾" in status and "Saving" in status:
                            status = "💾 Save..."
                        elif "📋" in status and "Loading" in status:
                            status = "📋 Load..."
                        else:
                            status = status[:max(1, col_width-4)] + "..."
                    row += f"{status:<{col_width}}"
                print(row)
            
            # Aggregate status information
            total_tasks = len(self.models) * len(display_apps_all)
            completed = 0
            failed = 0
            generating = 0
            
            for model in self.models:
                for app in display_apps_all:
                    status = self.status_matrix[model][app]
                    if "✅" in status or "Done" in status:
                        completed += 1
                    elif "❌" in status or "Failed" in status:
                        failed += 1
                    elif "✏️" in status or "Generating" in status:
                        generating += 1
            
            waiting = total_tasks - completed - failed - generating
            
            print("-" * self._term_cols)
            print(f"📊 Status: ✅ {completed} | ❌ {failed} | ✏️ {generating} | ⏳ {waiting} | Total: {total_tasks}")
            
            if total_tasks > 0:
                progress_percent = (completed + failed) / total_tasks * 100
                # Dynamic progress bar width while reserving text area
                pb_width = max(10, min(40, self._term_cols - 20))
                progress_bar = self._create_progress_bar(progress_percent, width=pb_width)
                print(f"📈 Progress: {progress_bar} {progress_percent:.1f}%")
                
                # ETA estimation
                if completed > 0 and elapsed > 0:
                    avg_time_per_task = elapsed / completed
                    remaining_tasks = total_tasks - completed - failed
                    eta_seconds = remaining_tasks * avg_time_per_task
                    eta_str = f"{int(eta_seconds//60):02d}:{int(eta_seconds%60):02d}"
                    print(f"⏱️ ETA: {eta_str} (avg {avg_time_per_task:.1f}s/task)")
            
            # Render detailed error, retry, analysis, and timing sections
            self._display_errors()
            self._display_retry_details()
            self._display_analysis_info()
            self._display_timing_info()
    
    def _create_progress_bar(self, percent: float, width: int = 40) -> str:
        """Create a textual progress bar."""
        filled = int(width * percent / 100)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"
    
    def _display_errors(self):
        """Display detailed error information."""
        errors = []
        
        for model in self.models:
            for app in self.apps:
                if app == "BATCH":
                    continue
                if self.error_details[model][app]:
                    errors.append({
                        'model': model,
                        'app': app,
                        'error': self.error_details[model][app]
                    })
        
        if errors:
            print("\n" + "━" * 16 + " ERRORS " + "━" * 16)
            max_show = 6 if self._small_screen else len(errors)
            hidden = max(0, len(errors) - max_show)
            show_list = errors[-max_show:] if max_show < len(errors) else errors
            if hidden > 0:
                print(f"❌ ... ({hidden} earlier errors hidden)")
            for error_info in show_list:
                print(f"❌ {error_info['app']} + {error_info['model']}:")
                # Show full error text
                error_text = error_info['error']
                print(f"   {error_text}")
            print("━" * 40)
    
    def _display_analysis_info(self):
        """Display recent analysis entries."""
        analysis_entries = []
        
        for model in self.models:
            for app in self.apps:
                if app == "BATCH":
                    continue
                if self.analysis_info[model][app]:
                    analysis_entries.extend([
                        f"{app} + {model}: {analysis}" 
                        for analysis in self.analysis_info[model][app]
                    ])
        
        if analysis_entries:
            print("\n" + "━" * 10 + " RECENT ANALYSIS " + "━" * 10)
            limit = 8 if self._small_screen else 15
            recent_entries = analysis_entries[-limit:] if len(analysis_entries) > limit else analysis_entries
            if len(analysis_entries) > limit:
                print(f"💬 ... ({len(analysis_entries) - limit} earlier analyses hidden)")
            for analysis in recent_entries:
                print(f"💬 {analysis}")
            print("━" * 35)
    
    def _display_retry_details(self):
        """Display retry details, with emphasis on GPT-5 related retries."""
        retry_info = []
        
        for model in self.models:
            for app in self.apps:
                if app == "BATCH":
                    continue
                if self.retry_details[model][app]:
                    retry_info.append({
                        'model': model,
                        'app': app,
                        'details': self.retry_details[model][app]
                    })
        
        if retry_info:
            print("\n" + "━" * 10 + " RETRY DETAILS " + "━" * 10)
            max_blocks = 4 if self._small_screen else len(retry_info)
            hidden_blocks = max(0, len(retry_info) - max_blocks)
            show_retry = retry_info[-max_blocks:] if max_blocks < len(retry_info) else retry_info
            if hidden_blocks > 0:
                print(f"🔄 ... ({hidden_blocks} earlier retry blocks hidden)")
            for info in show_retry:
                model = info['model']
                app = info['app']
                details = info['details']
                
                print(f"🔄 {app} + {model}:")
                
                if isinstance(details, list):  # Multiple retry attempts
                    limit = 4 if self._small_screen else len(details)
                    for attempt_info in details[-limit:]:
                        attempt_num = attempt_info.get('attempt', '?')
                        gen_time = attempt_info.get('generation_time', 0)
                        html_len = attempt_info.get('html_length', 0)
                        success = attempt_info.get('success', False)
                        
                        status_icon = "✅" if success else "❌"
                        print(f"   {status_icon} Attempt {attempt_num}: {gen_time}s → {html_len} chars")
                        
                        if 'validation' in attempt_info and attempt_info['validation']:
                            validation = attempt_info['validation']
                            if not validation.get('is_valid', True):
                                issues = validation.get('issues', [])
                                print(f"      Issues: {', '.join(issues[:2])}{'...' if len(issues) > 2 else ''}")
                        
                        if 'error' in attempt_info:
                            error = attempt_info['error'][:50] + "..." if len(attempt_info['error']) > 50 else attempt_info['error']
                            print(f"      Error: {error}")
                else:
                    # Single attempt or summary info
                    print(f"   Details: {str(details)[:100]}")
            print("━" * 35)
    
    def get_summary(self) -> Dict[str, int]:
        """Return current status counts."""
        total = len(self.models) * len(self.apps)
        completed = 0
        failed = 0
        
        for model in self.models:
            for app in self.apps:
                status = self.status_matrix[model][app]
                if "✅" in status:
                    completed += 1
                elif "❌" in status:
                    failed += 1
        
        return {
            'total': total,
            'completed': completed,
            'failed': failed,
            'running': total - completed - failed
        }
    
    def get_all_errors(self) -> List[Dict[str, str]]:
        """Return all error entries for writing into summary files."""
        errors = []
        
        for model in self.models:
            for app in self.apps:
                if self.error_details[model][app]:
                    errors.append({
                        'model': model,
                        'app': app,
                        'error': self.error_details[model][app],
                        'status': self.status_matrix[model][app]
                    })
        
        return errors
    
    def _display_timing_info(self):
        """Display timing and error-related timing entries (errors first)."""
        timing_entries = []
        error_entries = []
        
        for model in self.models:
            for app in self.apps:
                if self.timing_info[model][app]:
                    for timing in self.timing_info[model][app]:
                        # Split error vs regular timing messages
                        if ("failed:" in timing.lower() or "error:" in timing.lower() or 
                            "exception:" in timing.lower() or "FAILED:" in timing or 
                            "ERROR:" in timing or "EXCEPTION:" in timing):
                            # Truncate very long error timing messages
                            if len(timing) > 150:
                                timing = timing[:147] + "..."
                            error_entries.append(f"{app} + {model}: {timing}")
                        else:
                            timing_entries.append(f"{app} + {model}: {timing}")
        
        # First show error-related timing entries
        if error_entries:
            print("\n" + "━" * 10 + " ERRORS/FAILURES " + "━" * 10)
            elimit = 10 if self._small_screen else 15
            recent_errors = error_entries[-elimit:] if len(error_entries) > elimit else error_entries
            if len(error_entries) > elimit:
                print(f"❌ ... ({len(error_entries) - elimit} earlier errors hidden)")
            for error in recent_errors:
                print(f"❌ {error}")
        
        # Then show regular timing entries
        if timing_entries:
            print("\n" + "━" * 10 + " TIMING INFO " + "━" * 10)
            tlimit = 12 if self._small_screen else 20
            recent_entries = timing_entries[-tlimit:] if len(timing_entries) > tlimit else timing_entries
            if len(timing_entries) > tlimit:
                print(f"⏱️ ... ({len(timing_entries) - tlimit} earlier entries hidden)")
            for timing in recent_entries:
                print(f"⏱️ {timing}")
        
        if error_entries or timing_entries:
            print("━" * 35)
