#!/usr/bin/env python3
"""
Conjoint Analysis Script for Moral Machine Experiment

This script performs conjoint analysis (AMCE - Average Marginal Component Effect)
on decision data to determine the relevance of various factors (gender, age, 
species, social value, etc.) on decision-making.

The AMCE represents the average change in probability of being chosen when 
switching from one attribute level to another.

Usage:
    python scripts/conjoint_analysis.py --decision_run_name no_reasoning --results_dir data/results
"""

import sys
import os
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from moral_lens.config import PathConfig


class ConjointAnalyzer:
    """
    Performs conjoint analysis (AMCE) on decision data.
    
    The analysis leverages the experimental design where each dilemma presents
    two choices that differ along specific attribute dimensions. The data already
    contains phenomenon_category (the attribute being tested), category1, and 
    category2 columns that identify which attribute levels are being compared.
    """
    
    def __init__(self, results_dir: str = "data/results"):
        self.path_config = PathConfig(results_dir=results_dir)
    
    def find_result_files(self, decision_run_name: str) -> List[Path]:
        """Find all result files matching the decision_run_name pattern."""
        responses_dir = self.path_config.responses_output_dir
        
        # Pattern: {model_id}_{decision_run_name}_{i}.csv or {model_id}_{decision_run_name}.csv
        # Use regex to match exactly: _{decision_run_name}_\d+ or _{decision_run_name}.csv
        # This ensures we don't match variants like "no_reasoning_acted"
        all_csv_files = list(responses_dir.glob("*.csv"))
        files = []
        
        # Pattern 1: _{decision_run_name}_{number}.csv
        pattern_with_number = re.compile(rf"_{re.escape(decision_run_name)}_\d+\.csv$")
        # Pattern 2: _{decision_run_name}.csv (no number suffix)
        pattern_no_number = re.compile(rf"_{re.escape(decision_run_name)}\.csv$")
        
        for file in all_csv_files:
            file_str = file.name
            if pattern_with_number.search(file_str) or pattern_no_number.search(file_str):
                files.append(file)
        
        # Remove duplicates
        files = list(set(files))
        
        if not files:
            raise FileNotFoundError(
                f"No result files found matching pattern '{decision_run_name}' "
                f"in {responses_dir}"
            )
        
        return sorted(files)
    
    def load_data(self, decision_run_name: str) -> pd.DataFrame:
        """Load and combine all result files for a given run."""
        files = self.find_result_files(decision_run_name)
        print(f"Found {len(files)} result file(s):")
        for f in files:
            print(f"  - {f.name}")
        
        dfs = []
        for file in files:
            df = pd.read_csv(file, keep_default_na=False)
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"\nLoaded {len(combined_df)} total rows")
        
        return combined_df
    
    def compute_amce(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Compute Average Marginal Component Effects (AMCE).
        
        For each phenomenon category (Species, Gender, Age, etc.), we compute:
        - The probability of choosing each category level when presented
        - The AMCE: difference in choice probability between levels
        
        The data structure is:
        - phenomenon_category: the attribute being tested (e.g., "Species", "Gender")
        - category1: attribute level of choice1 (e.g., "Animals", "Male")
        - category2: attribute level of choice2 (e.g., "Humans", "Female")
        - decision_category: which category was actually chosen
        """
        # Filter out rows with empty decisions
        df = df[df['decision'].notna() & (df['decision'] != '')].copy()
        
        if len(df) == 0:
            raise ValueError("No valid decisions found in the data")
        
        # Initialize results
        results = []
        stats = {}
        
        # Get unique phenomenon categories
        phenomenon_categories = df['phenomenon_category'].unique()
        
        for phenomenon in phenomenon_categories:
            subset = df[df['phenomenon_category'] == phenomenon]
            n_trials = len(subset)
            
            if n_trials == 0:
                continue
            
            # Get all unique category levels for this phenomenon
            all_levels = set(subset['category1'].unique()) | set(subset['category2'].unique())
            
            # For each category level, compute how often it was chosen when presented
            level_stats = {}
            for level in all_levels:
                # Trials where this level was an option (either as choice1 or choice2)
                trials_with_level = subset[
                    (subset['category1'] == level) | (subset['category2'] == level)
                ]
                n_presented = len(trials_with_level)
                
                if n_presented == 0:
                    continue
                
                # Times this level was chosen
                n_chosen = len(trials_with_level[trials_with_level['decision_category'] == level])
                
                # Choice probability
                prob_chosen = n_chosen / n_presented
                
                # Standard error (binomial)
                se = np.sqrt(prob_chosen * (1 - prob_chosen) / n_presented)
                
                level_stats[level] = {
                    'n_presented': n_presented,
                    'n_chosen': n_chosen,
                    'prob_chosen': prob_chosen,
                    'se': se
                }
                
                results.append({
                    'phenomenon': phenomenon,
                    'level': level,
                    'n_presented': n_presented,
                    'n_chosen': n_chosen,
                    'prob_chosen': prob_chosen,
                    'se': se
                })
            
            # Compute AMCE (difference from baseline)
            # The baseline is typically the first level alphabetically or a "neutral" category
            if level_stats:
                sorted_levels = sorted(level_stats.keys())
                baseline = sorted_levels[0]
                baseline_prob = level_stats[baseline]['prob_chosen']
                
                stats[phenomenon] = {
                    'n_trials': n_trials,
                    'levels': level_stats,
                    'baseline': baseline,
                    'baseline_prob': baseline_prob
                }
        
        results_df = pd.DataFrame(results)
        return results_df, stats
    
    def analyze(self, decision_run_name: str) -> Dict:
        """Perform conjoint analysis."""
        print(f"\n{'='*70}")
        print(f"Conjoint Analysis (AMCE) for: {decision_run_name}")
        print(f"{'='*70}\n")
        
        # Load data
        df = self.load_data(decision_run_name)
        
        # Compute AMCE
        results_df, stats = self.compute_amce(df)
        
        # Valid decisions
        valid_df = df[df['decision'].notna() & (df['decision'] != '')]
        total_valid = len(valid_df)
        
        print(f"\nTotal valid decisions: {total_valid}")
        print(f"Unique phenomena tested: {len(stats)}")
        
        # Print results by phenomenon category
        print(f"\n{'='*70}")
        print("AMCE RESULTS BY ATTRIBUTE")
        print(f"{'='*70}")
        
        # Sort phenomena by the range of effect sizes (most influential first)
        phenomenon_effects = {}
        for phenomenon, data in stats.items():
            probs = [v['prob_chosen'] for v in data['levels'].values()]
            effect_range = max(probs) - min(probs) if probs else 0
            phenomenon_effects[phenomenon] = effect_range
        
        sorted_phenomena = sorted(phenomenon_effects.keys(), 
                                  key=lambda x: phenomenon_effects[x], 
                                  reverse=True)
        
        for phenomenon in sorted_phenomena:
            data = stats[phenomenon]
            print(f"\n{phenomenon}")
            print(f"  Total trials: {data['n_trials']}")
            print(f"  Baseline: {data['baseline']} (prob = {data['baseline_prob']:.3f})")
            print(f"  {'Level':<25} {'Prob':>8} {'AMCE':>10} {'95% CI':>18} {'N':>8}")
            print(f"  {'-'*65}")
            
            baseline_prob = data['baseline_prob']
            
            # Sort levels for consistent display
            sorted_levels = sorted(data['levels'].keys())
            
            for level in sorted_levels:
                level_data = data['levels'][level]
                prob = level_data['prob_chosen']
                se = level_data['se']
                amce = prob - baseline_prob
                ci_low = amce - 1.96 * se
                ci_high = amce + 1.96 * se
                n = level_data['n_presented']
                
                # Mark baseline with (base)
                level_str = f"{level} (base)" if level == data['baseline'] else level
                ci_str = f"[{ci_low:+.3f}, {ci_high:+.3f}]"
                
                print(f"  {level_str:<25} {prob:>8.3f} {amce:>+10.3f} {ci_str:>18} {n:>8}")
        
        # Summary table: Overall importance ranking
        print(f"\n{'='*70}")
        print("ATTRIBUTE IMPORTANCE RANKING")
        print("(Based on max probability difference between levels)")
        print(f"{'='*70}\n")
        
        print(f"{'Rank':<6} {'Attribute':<20} {'Effect Size':>12} {'Interpretation':<30}")
        print("-" * 70)
        
        for rank, phenomenon in enumerate(sorted_phenomena, 1):
            effect = phenomenon_effects[phenomenon]
            
            # Interpretation
            if effect >= 0.3:
                interp = "Very strong preference"
            elif effect >= 0.2:
                interp = "Strong preference"
            elif effect >= 0.1:
                interp = "Moderate preference"
            elif effect >= 0.05:
                interp = "Weak preference"
            else:
                interp = "Negligible effect"
            
            print(f"{rank:<6} {phenomenon:<20} {effect:>12.3f} {interp:<30}")
        
        # Print specific preference patterns
        print(f"\n{'='*70}")
        print("PREFERENCE PATTERNS")
        print(f"{'='*70}\n")
        
        for phenomenon in sorted_phenomena:
            data = stats[phenomenon]
            levels = data['levels']
            
            # Find most and least preferred
            sorted_by_prob = sorted(levels.items(), key=lambda x: x[1]['prob_chosen'], reverse=True)
            most_preferred = sorted_by_prob[0]
            least_preferred = sorted_by_prob[-1]
            
            diff = most_preferred[1]['prob_chosen'] - least_preferred[1]['prob_chosen']
            
            if diff >= 0.05:  # Only show if meaningful difference
                print(f"{phenomenon}: {most_preferred[0]} preferred over {least_preferred[0]}")
                print(f"  → {most_preferred[0]}: {most_preferred[1]['prob_chosen']:.1%} chosen when presented")
                print(f"  → {least_preferred[0]}: {least_preferred[1]['prob_chosen']:.1%} chosen when presented")
                print(f"  → Effect size: {diff:.3f} ({diff*100:.1f} percentage points)")
                print()
        
        return {
            'results_df': results_df,
            'stats': stats,
            'phenomenon_effects': phenomenon_effects,
            'total_valid': total_valid
        }


def main():
    parser = argparse.ArgumentParser(
        description='Perform conjoint analysis (AMCE) on decision data'
    )
    parser.add_argument(
        '--decision_run_name',
        type=str,
        required=True,
        help='Decision run name (without the _{i} suffix, e.g., "no_reasoning")'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='data/results',
        help='Directory containing results (default: data/results)'
    )
    
    args = parser.parse_args()
    
    analyzer = ConjointAnalyzer(results_dir=args.results_dir)
    analyzer.analyze(args.decision_run_name)
    
    print(f"\n{'='*70}")
    print("Analysis complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
