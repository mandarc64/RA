"""Profiling code for Jax and PyTorch.

Modified from:
https://github.com/Lightning-AI/lightning/tree/master/src/pytorch_lightning/profilers.
"""

import jax
import time
from jax.profiler import TraceAnnotation, start_trace, stop_trace
import os
from typing import Dict, Optional, Tuple, List
from collections import defaultdict
from contextlib import contextmanager
import numpy as np
import torch
import csv

def _get_time() -> float:
    """Returns the current time with synchronization for JAX devices."""
    # Ensure all pending operations on all devices have completed.
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()
    return time.monotonic()


class JAXProfiler:
    def __init__(self, local_rank: Optional[int] = None) -> None:
        self._local_rank = local_rank
        self.current_actions = []  # Stack of current actions to track hierarchy
        self.recorded_durations = defaultdict(lambda: {'total': 0, 'count': 0, 'children': defaultdict(int)})
        self.start_time = _get_time()
        self.trace_started = False

    @property
    def local_rank(self) -> int:
        return 0 if self._local_rank is None else self._local_rank

    def start(self, action_name: str) -> None:
        
        # This code checks if the profiler is being run on the main device (local rank 0). 
        # If not, it returns early without doing anything, as it's meant to profile only the primary device.
        if self.local_rank != 0:
            return
        
        # Gets the current time
        current_time = _get_time()

        # This checks if the action that's trying to be started has already been started
        if self.current_actions and action_name in self.current_actions[-1]:
            raise ValueError(f'Attempted to start {action_name} which has already started.')
        
        # Adds the current action and its start time to the stack of actions being profiled.
        self.current_actions.append({'name': action_name, 'start': current_time})
        
        # Starts the JAX trace if it hasn't been started yet, to collect more detailed profiling information, and marks trace_started as True.
        if not self.trace_started:
            start_trace("/scratch/mchaud21/algorithmic-efficiency")
            self.trace_started = True

    def stop(self) -> None:

        # Similar to start, this returns early if not on the primary device 
        # or if there are no actions to stop (i.e., the stack is empty).
        if self.local_rank != 0 or not self.current_actions:
            return
        
        # Gets the current time
        end_time = _get_time()

        # Pops the last action from the stack, 
        # calculates the duration by subtracting the start time from the current time, 
        # and retrieves the action name.
        current_action = self.current_actions.pop()
        duration = end_time - current_action['start']
        action_name = current_action['name']

        # Updates the recorded_durations dictionary with the total duration of the action
        # and increments its count.

        self.recorded_durations[action_name]['total'] += duration 
        self.recorded_durations[action_name]['count'] += 1 

        # If there are still actions on the stack (meaning the current action had a parent), 
        # it updates the parent action's children's total duration.

        # Assuming "Update parameters" is nested within "One Training Step", and the stop method is called for "Update parameters".
        if self.current_actions:
            
            # Assuming `current_actions` has 'One Training Step' as the last item
            parent_action = self.current_actions[-1]['name'] # This would be 'One Training Step' in the example above
            self.recorded_durations[parent_action]['children'][action_name] += duration # Adds the duration of 'Update parameters' to the children of 'One Training Step' in the example above
             # {
            #     'One Training Step': {
            #         'total': current_total_time_for_one_training_step, 
            #         'count': current_count_for_one_training_step, 
            #         'children': {
            #             'Update parameters': 855.58989
            #             # Other children of 'One Training Step'
            #         }
            #     }
            # }

        else:
           # If the current action is the top-level action (no other actions on the stack), 
            # it ensures there is a 'Total' entry in recorded_durations and updates the total duration.    
            if 'Total' not in self.recorded_durations:
                self.recorded_durations['Total'] = {'total': 0, 'count': 0, 'children': defaultdict(int)}
            self.recorded_durations['Total']['children'][action_name] += duration

        # If all actions have stopped, it updates the overall total duration and count.
        if not self.current_actions:
            self.recorded_durations['Total']['total'] += duration
            self.recorded_durations['Total']['count'] += 1

    @contextmanager
    def profile(self, action_name: str) -> None:
        self.start(action_name)
        try:
            with TraceAnnotation(action_name):
                yield
        finally:
            self.stop()

    def _make_report(self) -> str:
        def format_report(action, data, level=0):

            # Sets the indentation level based on how deep in the stack the action is.
            indent = '  ' * level

           # Calculates the total time spent on the action and its percentage of the overall time.
            total_time = data['total']
            percentage = 100.0 * total_time / self.recorded_durations['Total']['total'] if self.recorded_durations['Total']['total'] > 0 else 0
            
            # Formats the report string with indentation, action name, time, and percentage. 
            # It then recursively calls itself to format reports for any child actions.
            result = f'{indent}{action}: {total_time:.5f}s ({percentage:.2f}%)'
            for child_action, child_time in sorted(data['children'].items(), key=lambda item: item[1], reverse=True):
                child_data = self.recorded_durations[child_action]
                result += '\n' + format_report(child_action, child_data, level + 1)
            return result
        # This calls format_report for the top-level 'Total' action and returns the full formatted report.
        report = format_report('Total', self.recorded_durations['Total'])
        return report

    def summary(self) -> str:
        sep = os.linesep
        output_string = f'Profiler Report{sep}:' + self._make_report()
        if self.trace_started:
            stop_trace()
            self.trace_started = False
        return output_string
    
    def export_to_csv(self, file_name: str) -> None:
        with open(file_name, 'w', newline='') as csvfile:
            fieldnames = ['Action', 'Total Time (s)', 'Percentage', 'Count', 'Parent Action']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            def write_row(action, data, parent_action=''):
                total_time = data['total']
                percentage = 100.0 * total_time / self.recorded_durations['Total']['total'] if self.recorded_durations['Total']['total'] > 0 else 0
                writer.writerow({
                    'Action': action,
                    'Total Time (s)': total_time,
                    'Percentage': percentage,
                    'Count': data['count'],
                    'Parent Action': parent_action
                })
                for child_action in data['children']:
                    child_data = self.recorded_durations[child_action]
                    write_row(child_action, child_data, action)

            write_row('Total', self.recorded_durations['Total'])


class PassThroughProfiler(JAXProfiler):

    def start(self, action_name: str) -> None:
        pass

    def stop(self, action_name: str) -> None:
        pass
