#!/workspace/HOME/guest/.conda/envs/spar/bin/python

import argparse
import subprocess
import os
import sys
import threading
import queue
import time
import shutil
import atexit
import shlex
import collections # Import collections for deque

# --- Global state ---
# List of panes created by this script for cleanup
created_pane_ids = []
# Lock for safely appending to created_pane_ids list
pane_list_lock = threading.Lock()
# Flag to signal termination to workers
terminate_event = threading.Event()
# Store the initial pane ID where the script runs
initial_pane_id = None

# --- Cleanup Function ---
def cleanup():
    """Signals workers to terminate but leaves panes open."""
    # Only signal termination, do not kill panes
    print("\nSignaling worker threads to terminate...", file=sys.stderr)
    terminate_event.set() # Signal workers it's time to stop

    # Give workers a moment to react
    time.sleep(0.5)

    print("Cleanup finished. Panes will remain open.", file=sys.stderr)


# Register cleanup to run on script exit
atexit.register(cleanup)

# --- GPU Detection ---
def get_available_gpus():
    """Detects available GPU IDs using nvidia-smi."""
    nvidia_smi_path = shutil.which('nvidia-smi')
    if not nvidia_smi_path:
        print("Error: nvidia-smi command not found. Cannot automatically detect GPUs.", file=sys.stderr)
        sys.exit(1)
    try:
        result = subprocess.run(
            [nvidia_smi_path, '--query-gpu=index', '--format=csv,noheader'],
            capture_output=True, text=True, check=True, encoding='utf-8'
        )
        gpu_ids_str = result.stdout.strip().split('\n')
        gpu_ids = [int(idx) for idx in gpu_ids_str if idx]
        if not gpu_ids and gpu_ids_str != ['']: # Handle case where nvidia-smi returns empty string vs no GPUs
             print("Warning: nvidia-smi found, but no GPUs detected.", file=sys.stderr)
             # Allow script to continue if user explicitly specified GPUs later
        print(f"Detected GPUs: {gpu_ids}") # Print detection result
        return gpu_ids
    except FileNotFoundError:
        print("Error: nvidia-smi command not found.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi: {e}", file=sys.stderr)
        print(f"Stderr: {e.stderr}", file=sys.stderr)
        sys.exit(1)
    except ValueError:
        print(f"Error parsing GPU IDs from nvidia-smi output: {result.stdout}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during GPU detection: {e}", file=sys.stderr)
        sys.exit(1)


# --- Worker Function ---
def gpu_worker(gpu_id, job_queue, ready_gpu_queue, session_name, pane_id):
    """
    Worker function executed in a thread for each GPU.
    Signals readiness, gets assigned jobs from job_queue, sends commands, signals readiness again.
    """
    print(f"Worker GPU {gpu_id} (Pane {pane_id}): Started.")
    # Signal initial readiness
    print(f"Worker GPU {gpu_id} (Pane {pane_id}): Signaling initial readiness.")
    ready_gpu_queue.put(gpu_id)

    job_count = 0
    while not terminate_event.is_set():
        job_command = None
        try:
            # Wait indefinitely for a job assigned by the main thread via job_queue
            print(f"Worker GPU {gpu_id} (Pane {pane_id}): Waiting for assigned job...")
            # Use timeout to allow checking terminate_event periodically
            job_command = job_queue.get(block=True, timeout=1)
        except queue.Empty:
             if terminate_event.is_set():
                 print(f"Worker GPU {gpu_id} (Pane {pane_id}): Termination signal received while waiting for job.")
                 break # Exit loop if terminating
             continue # Keep waiting if not terminating

        # Process the received item
        if job_command is None: # Sentinel value indicates no more jobs
            print(f"Worker GPU {gpu_id} (Pane {pane_id}): Received sentinel. Signaling done.")
            job_queue.task_done() # Mark the sentinel as done
            # DO NOT put gpu_id back on ready queue here, main thread handles shutdown signaling
            break # Exit the worker loop

        # If it's a real job command
        job_count += 1
        print(f"Worker GPU {gpu_id} (Pane {pane_id}): Received assigned job {job_count}: '{job_command}'")

        # 3. Prepare the command sequence
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        command_in_subshell = f"{job_command}"
        safe_command_in_subshell = shlex.quote(command_in_subshell)
        full_command_sequence = [
            f"export CUDA_VISIBLE_DEVICES={gpu_id}",
            f"echo",
            f"echo '--- [{timestamp}] Starting Job {job_count} on GPU {gpu_id} (Pane {pane_id}) ---'",
            f"echo 'Command: {job_command}'",
            f"echo '----------------------------------------'",
            f"bash -c {safe_command_in_subshell}",
            f"EXIT_CODE=$?",
            f"echo '----------------------------------------'",
            f"echo '--- Job {job_count} on GPU {gpu_id} Finished (Exit Code: $EXIT_CODE) ---'",
            f"echo ''"
        ]
        tmux_pane_command_sequence = "; ".join(full_command_sequence)

        # 4. Send the command sequence to the assigned tmux pane
        tmux_cmd = ['tmux', 'send-keys', '-t', pane_id, tmux_pane_command_sequence, 'C-m']
        send_success = False
        try:
            # Check pane existence
            list_cmd = ['tmux', 'list-panes', '-F', '#{pane_id}']
            result = subprocess.run(list_cmd, capture_output=True, text=True, check=True)
            current_panes = result.stdout.strip().split('\n')
            if pane_id not in current_panes:
                 print(f"Worker GPU {gpu_id} (Pane {pane_id}): Target pane closed prematurely. Skipping job {job_count}.", file=sys.stderr)
                 job_queue.task_done() # Still need to mark task done for the job we got
                 print(f"Worker GPU {gpu_id} (Pane {pane_id}): Marked skipped job {job_count} as done.")
                 # Don't signal readiness here, wait for next assigned job or sentinel
                 continue

            # Send the command
            job_send_timeout = 60
            print(f"Worker GPU {gpu_id} (Pane {pane_id}): Sending job {job_count} command (timeout {job_send_timeout}s)...")
            subprocess.run(tmux_cmd, check=True, capture_output=True, text=True, timeout=job_send_timeout)
            print(f"Worker GPU {gpu_id} (Pane {pane_id}): Successfully sent job {job_count} command.")
            send_success = True
        except subprocess.TimeoutExpired:
             print(f"Worker GPU {gpu_id} (Pane {pane_id}): Timeout sending job {job_count} command.", file=sys.stderr)
        except subprocess.CalledProcessError as e:
            if "no such pane" in e.stderr or "session not found" in e.stderr:
                 print(f"Worker GPU {gpu_id} (Pane {pane_id}): Target pane likely closed just before send. Skipping job {job_count}.", file=sys.stderr)
            else:
                print(f"Worker GPU {gpu_id} (Pane {pane_id}): Error sending job {job_count} command: {e.stderr}", file=sys.stderr)
        except FileNotFoundError:
             print(f"Worker GPU {gpu_id} (Pane {pane_id}): tmux command not found. Terminating.", file=sys.stderr)
             terminate_event.set()
             job_queue.task_done() # Mark current job done before breaking
             break
        except Exception as e_inner:
             print(f"Worker GPU {gpu_id} (Pane {pane_id}): Unexpected error sending job {job_count}: {e_inner}", file=sys.stderr)

        # Mark task done *before* signaling readiness for next job
        print(f"Worker GPU {gpu_id} (Pane {pane_id}): Marking job {job_count} as done in queue (Send success: {send_success}).")
        job_queue.task_done()

        # Signal readiness for the next job AFTER completing the previous one
        if not terminate_event.is_set():
             print(f"Worker GPU {gpu_id} (Pane {pane_id}): Signaling readiness for next job.")
             ready_gpu_queue.put(gpu_id) # Put self back onto the ready queue
        else:
             print(f"Worker GPU {gpu_id} (Pane {pane_id}): Termination signaled, not signaling readiness.")


    # --- Worker Exit ---
    print(f"Worker GPU {gpu_id} (Pane {pane_id}): Exiting run loop.")
    final_msg = ""
    if not terminate_event.is_set(): # Normal exit (finished queue)
        print(f"Worker GPU {gpu_id}: Finished processing jobs for pane {pane_id}.")
        final_msg = f"echo ''; echo '--- Worker GPU {gpu_id} (Pane {pane_id}): All assigned jobs complete. Processed {job_count} jobs. ---'"
    else: # Terminated early
         print(f"Worker GPU {gpu_id}: Terminating early for pane {pane_id}.")
         final_msg = f"echo ''; echo '--- Worker GPU {gpu_id} (Pane {pane_id}): Terminated by script. Processed {job_count} jobs. ---'"

    if final_msg:
        safe_final_msg = shlex.quote(final_msg)
        tmux_final_cmd = ['tmux', 'send-keys', '-t', pane_id, f"bash -c {safe_final_msg}", 'C-m']
        try:
            # Check if pane still exists before sending final message
            list_cmd = ['tmux', 'list-panes', '-F', '#{pane_id}']
            result = subprocess.run(list_cmd, capture_output=True, text=True, check=True)
            current_panes = result.stdout.strip().split('\n')
            if pane_id in current_panes:
                subprocess.run(tmux_final_cmd, check=False, capture_output=True, timeout=5)
        except subprocess.TimeoutExpired:
             print(f"Worker GPU {gpu_id}: Warning - timeout sending final message to tmux pane {pane_id}", file=sys.stderr)
        except Exception as e_send:
             # Ignore errors sending final message if pane is already gone
             if not ("no such pane" in str(e_send) or "session not found" in str(e_send)):
                 print(f"Worker GPU {gpu_id}: Warning - could not send final message to tmux pane {pane_id}: {e_send}", file=sys.stderr)

    print(f"Worker thread for GPU {gpu_id} (pane {pane_id}) stopped.")


# --- Main Entry Point ---
if __name__ == "__main__":
    # --- Check if running inside tmux ---
    if not os.getenv('TMUX'):
        print("Error: This script must be run from within a tmux session.", file=sys.stderr)
        sys.exit(1)

    # --- Get current pane ID ---
    try:
        result = subprocess.run(['tmux', 'display-message', '-p', '#{pane_id}'], check=True, capture_output=True, text=True)
        initial_pane_id = result.stdout.strip()
        print(f"Running dispatcher in tmux pane: {initial_pane_id}")
    except Exception as e:
        print(f"Error getting current tmux pane ID: {e}", file=sys.stderr)
        sys.exit(1)


    parser = argparse.ArgumentParser(
        description="Dispatch shell commands to GPUs using tmux panes. MUST BE RUN INSIDE TMUX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("job_script_file", help="Path to job script file.")
    parser.add_argument("--gpus", help="Comma-separated GPU IDs (e.g., 0,1). Defaults to all.")
    # Session name is less relevant, maybe remove or make optional? Keep for worker context for now.
    parser.add_argument("--session-name", default=os.getenv('TMUX', '').split(',')[0], help="Tmux session name (used for worker context messages). Defaults to current session.")
    parser.add_argument("--nvtop", action="store_true", help="Include an nvtop pane.")
    # REMOVED internal flags: --run-in-pane, --worker-pane-ids

    args = parser.parse_args()

    # --- Check external commands ---
    if not shutil.which('tmux'):
        # Should not happen due to initial check, but good practice
        print("Error: tmux command not found.", file=sys.stderr)
        sys.exit(1)
    if args.nvtop and not shutil.which('nvtop'):
         print("Warning: --nvtop specified, but nvtop command not found. Skipping nvtop pane.", file=sys.stderr)
         args.nvtop = False

    # --- Determine GPUs ---
    if args.gpus:
        try:
            gpu_ids_str = [g.strip() for g in args.gpus.split(',') if g.strip()]
            if not gpu_ids_str: raise ValueError("No GPU IDs provided.")
            gpu_ids = [int(g) for g in gpu_ids_str]
            print(f"Using specified GPUs: {gpu_ids}")
        except ValueError as e:
            print(f"Error: Invalid GPU IDs '{args.gpus}'. {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("No --gpus specified, attempting to detect automatically...")
        gpu_ids = get_available_gpus()

    if not gpu_ids:
        print("Error: No GPUs available or specified.", file=sys.stderr)
        sys.exit(1)

    # --- Create Panes ---
    num_gpus = len(gpu_ids)
    worker_pane_ids = {} # Map gpu_id to pane_id (e.g., '%1')
    nvtop_pane_id = None

    # Create nvtop pane (if requested) by splitting the *current* pane
    if args.nvtop:
        try:
            print("Creating nvtop pane...")
            # Split current pane vertically, run nvtop, capture new pane ID
            split_cmd = ['tmux', 'split-window', '-v', '-t', initial_pane_id, '-P', '-F', '#{pane_id}', 'nvtop']
            result = subprocess.run(split_cmd, check=True, capture_output=True, text=True)
            nvtop_pane_id = result.stdout.strip()
            print(f"nvtop started in pane ID {nvtop_pane_id}")
            with pane_list_lock:
                created_pane_ids.append(nvtop_pane_id) # Track for cleanup
            time.sleep(0.3)
        except Exception as e:
            print(f"Error creating or running nvtop in pane: {e}", file=sys.stderr)
            # Continue without nvtop pane

    # Create worker panes by splitting the *current* pane
    print(f"Creating {num_gpus} panes for GPU workers...")
    for i in range(num_gpus):
        gpu_id = gpu_ids[i]
        try:
            # Split current pane horizontally, capture new pane ID
            split_cmd = ['tmux', 'split-window', '-h', '-t', initial_pane_id, '-P', '-F', '#{pane_id}']
            result = subprocess.run(split_cmd, check=True, capture_output=True, text=True)
            new_pane_id = result.stdout.strip()
            worker_pane_ids[gpu_id] = new_pane_id
            print(f"  GPU {gpu_id} assigned to pane ID {new_pane_id}")
            with pane_list_lock:
                created_pane_ids.append(new_pane_id) # Track for cleanup
            time.sleep(0.3)
        except Exception as e:
            print(f"Error creating pane for GPU {gpu_id}: {e}", file=sys.stderr)
            # Cleanup will run via atexit
            sys.exit(1)

    # Apply layout to the current window
    try:
        print("Applying tiled layout...")
        # Target the current window using '.'
        subprocess.run(['tmux', 'select-layout', '-t', '.', 'tiled'], check=True)
        time.sleep(0.1)
    except Exception as e:
        print(f"Warning: Failed to apply tiled layout: {e}", file=sys.stderr)


    # --- Run Dispatcher Logic ---
    print("-" * 30)
    print("--- Dispatcher Started ---")
    print(f"Session: {args.session_name}") # Use session name from args/env
    print(f"Job File: {args.job_script_file}")
    print(f"GPUs: {gpu_ids}")
    print(f"Worker Pane IDs: {worker_pane_ids}")
    print("-" * 30)

    # --- Read Jobs ---
    try:
        with open(args.job_script_file, 'r') as f:
            jobs = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        if not jobs:
            print(f"Error: No jobs found in {args.job_script_file}", file=sys.stderr)
            sys.exit(1)
        total_jobs = len(jobs) # Store total number
        print(f"Found {total_jobs} jobs to dispatch across {num_gpus} GPUs.")
    except FileNotFoundError:
        print(f"Error: Job script file not found: {args.job_script_file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading job script file: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Setup Queues and Worker Threads ---
    job_queue = queue.Queue() # For main -> worker job assignment
    ready_gpu_queue = queue.Queue() # For worker -> main readiness signal
    threads = []

    print("Starting GPU worker threads...") # Log thread start initiation
    for gpu_id in gpu_ids:
        if gpu_id not in worker_pane_ids:
             print(f"Error: No pane assigned for GPU {gpu_id}. Skipping worker.", file=sys.stderr)
             continue
        pane_id = worker_pane_ids[gpu_id]
        print(f"  Preparing thread for GPU {gpu_id} (Pane ID {pane_id})...")
        # Pass ready_gpu_queue to worker
        thread = threading.Thread(target=gpu_worker, args=(gpu_id, job_queue, ready_gpu_queue, args.session_name, pane_id), daemon=True)
        threads.append(thread)
        thread.start()
        print(f"  Thread for GPU {gpu_id} started.")

    print("All worker threads started. Waiting for initial readiness signals...")
    # Optional: Wait for all workers to signal initial readiness for robustness
    initial_ready_count = 0
    ready_gpus_initial = set()
    initial_wait_start = time.time()
    initial_wait_timeout = 30 # seconds
    # Use num_gpus derived from the actual threads started
    num_started_threads = len(threads)
    while initial_ready_count < num_started_threads and time.time() - initial_wait_start < initial_wait_timeout:
        try:
             # Get from ready queue with timeout
             gpu_id = ready_gpu_queue.get(timeout=1)
             if gpu_id not in ready_gpus_initial:
                 print(f"  Initial readiness signal received from GPU {gpu_id}.")
                 ready_gpus_initial.add(gpu_id)
                 initial_ready_count += 1
             else:
                 # This worker signaled ready again before dispatch started, put it back
                 ready_gpu_queue.put(gpu_id)
                 time.sleep(0.1) # Avoid busy loop if something is wrong
        except queue.Empty:
             # Timeout, check main timeout condition
             if time.time() - initial_wait_start >= initial_wait_timeout:
                 break # Exit loop if overall timeout exceeded
             # Otherwise continue waiting
             pass
    if initial_ready_count < num_started_threads:
         print(f"Warning: Only {initial_ready_count}/{num_started_threads} workers signaled initial readiness within {initial_wait_timeout}s.", file=sys.stderr)
         # Consider exiting if this happens? For now, proceed.

    # --- Dispatch Jobs Sequentially Based on Readiness ---
    print(f"\n--- Starting Job Dispatch (Total: {total_jobs}) ---")
    jobs_dispatched_count = 0
    # Use a deque for efficient removal from the front
    jobs_to_do_deque = collections.deque(jobs)

    # Loop while there are jobs left to dispatch
    while jobs_dispatched_count < total_jobs:
        if terminate_event.is_set():
             print("Termination signaled during job dispatch.", file=sys.stderr)
             break # Exit dispatch loop if termination requested

        # Wait for the next available worker
        print(f"Waiting for next available worker (Ready queue size: {ready_gpu_queue.qsize()})...")
        try:
            # Block until a worker signals readiness, use timeout to check terminate_event
            gpu_id = ready_gpu_queue.get(block=True, timeout=1)
        except queue.Empty:
            # No worker ready, loop again and check terminate_event
            continue

        # Check if there are still jobs to assign (should be true if loop condition is correct)
        if jobs_to_do_deque:
            # Get the next job
            job_command = jobs_to_do_deque.popleft()
            jobs_dispatched_count += 1
            print(f"Dispatching job {jobs_dispatched_count}/{total_jobs} ('{job_command}') to ready GPU {gpu_id}.")
            # Put the job onto the main job queue for the worker to pick up
            job_queue.put(job_command)
        else:
            # Should not happen if loop condition is jobs_dispatched_count < total_jobs
            # If it does, put the worker back onto the ready queue and break
            print("Warning: Worker ready but no jobs left in deque (should not happen).", file=sys.stderr)
            ready_gpu_queue.put(gpu_id)
            break

    # Check if dispatch was terminated early
    if jobs_dispatched_count == total_jobs:
        print("--- All jobs dispatched to workers. ---")
    else:
        print(f"--- Dispatch loop finished early ({jobs_dispatched_count}/{total_jobs} dispatched). ---")


    # --- Signal Workers to Finish ---
    print("Dispatching sentinels to job queue for workers to exit...")
    sentinels_dispatched_count = 0
    # Send one sentinel for each worker thread that was started
    for _ in threads:
        job_queue.put(None)
        sentinels_dispatched_count += 1
    print(f"{sentinels_dispatched_count} sentinels placed on job queue.")

    # --- Wait for Workers ---
    # Wait for job_queue to be fully processed (all jobs + sentinels marked done)
    print(f"Waiting for workers to process remaining jobs and sentinels (job_queue.join())...")
    try:
        job_queue.join() # Blocks until task_done() called for all items
        print("job_queue.join() completed.")
    except KeyboardInterrupt:
         print("\n--- Interrupted while waiting for final processing ---", file=sys.stderr)
         terminate_event.set() # Ensure workers are signaled
         print("job_queue.join() interrupted.")
         # Give workers time to react to terminate_event before joining threads
         time.sleep(1)
    except Exception as e:
        print(f"Unexpected error during final job_queue.join(): {e}", file=sys.stderr)
        terminate_event.set() # Ensure workers are signaled
        print("job_queue.join() failed due to error.")

    # --- Join Threads ---
    print("Waiting for worker threads to stop (thread.join())...")
    all_threads_joined = True
    for thread in threads:
        print(f"  Joining thread {thread.name}...")
        thread.join(timeout=15)
        if thread.is_alive():
             print(f"Warning: Worker thread {thread.name} did not terminate cleanly after job queue join.", file=sys.stderr)
             all_threads_joined = False
        else:
             print(f"  Thread {thread.name} joined.")

    if all_threads_joined:
        print("All worker threads have stopped.")
    else:
        print("Some worker threads did not stop cleanly.")

    print("-" * 30)
    print("--- Dispatcher Finished ---")
    # Cleanup (signaling) runs automatically via atexit

    sys.exit(0) # Explicitly exit normally