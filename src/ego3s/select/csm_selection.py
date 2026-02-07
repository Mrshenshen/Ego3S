#!/usr/bin/env python3

import json
import sys
import os
from pathlib import Path

DEBUG = False

def calculate_mean_reward(entry):
    format_rewards = entry.get('format_rewards', [])
    accuracy_rewards = entry.get('accuracy_rewards', [])
    
    # if len(format_rewards) != len(accuracy_rewards):
    #     print(f"Warning: Length mismatch in problem_id {entry.get('problem_id', 'unknown')}: "
    #           f"format_rewards={len(format_rewards)}, accuracy_rewards={len(accuracy_rewards)}")
    
    total_sum = sum(format_rewards) + sum(accuracy_rewards)
    total_count = len(format_rewards) + len(accuracy_rewards)
    
    if total_count == 0:
        return 0.0
    
    return total_sum / total_count


def calculate_rewards(input_path, output_path=None):
    # print(f"[Calculate] Reading {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        data = [data]
        single_object = True
    else:
        single_object = False
    
    print(f"[Calculate] Processing {len(data)} entries...")
    
    for i, entry in enumerate(data):
        mean_reward = calculate_mean_reward(entry)
        entry['mean_total_reward'] = mean_reward
        
        # if i < 5 or i % 1000 == 0:
        #     print(f"  Entry {i} (problem_id: {entry.get('problem_id', 'N/A')}): "
        #           f"mean_total_reward = {mean_reward:.6f}")
    
    rewards = [entry['mean_total_reward'] for entry in data]
    print(f"\n[Calculate] Statistics:")
    print(f"  Total entries: {len(rewards)}")
    print(f"  Min reward: {min(rewards):.6f}")
    print(f"  Max reward: {max(rewards):.6f}")
    print(f"  Mean reward: {sum(rewards)/len(rewards):.6f}")
    
    if output_path is None:
        output_path = input_path
    
    # print(f"\n[Calculate] Writing to {output_path}...")
    
    output_data = data[0] if single_object else data
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print("[Calculate] Done!")


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def filter_by_reward_diff(text_file, video_file, output_dir, threshold):
    # print(f"[Filter] Loading data files...")
    text_data = load_json(text_file)
    video_data = load_json(video_file)
    
    text_dict = {item['problem_id']: item for item in text_data}
    
    matched_text_rewards = []
    matched_video_rewards = []
    
    for video_item in video_data:
        problem_id = video_item['problem_id']
        if problem_id in text_dict:
            matched_text_rewards.append(text_dict[problem_id]['mean_total_reward'])
            matched_video_rewards.append(video_item['mean_total_reward'])
    
    text_avg = sum(matched_text_rewards) / len(matched_text_rewards) if matched_text_rewards else 0
    video_avg = sum(matched_video_rewards) / len(matched_video_rewards) if matched_video_rewards else 0
    avg_diff = video_avg - text_avg
    
    print(f"[Filter] Average rewards:")
    print(f"  Text: {text_avg:.6f}")
    print(f"  Video: {video_avg:.6f}")
    print(f"  Difference: {avg_diff:.6f}")
    print(f"  Threshold: {threshold}")
    
    filtered_text_data = []
    filtered_video_data = []
    filtered_text_freeform = []
    filtered_video_freeform = []
    filtered_text_multiple = []
    filtered_video_multiple = []
    
    not_found_count = 0
    freeform_count = 0
    multiple_count = 0
    
    for video_item in video_data:
        problem_id = video_item['problem_id']
        problem_type = video_item.get('problem_type', 'unknown')
        
        if problem_id not in text_dict:
            not_found_count += 1
            # print(f"  Warning: problem_id {problem_id} not found in text file")
            continue
        
        text_item = text_dict[problem_id]
        
        text_reward = text_item['mean_total_reward']
        video_reward = video_item['mean_total_reward']
        item_diff = video_reward - text_reward
        
        deviation = abs(item_diff - avg_diff)
        
        if deviation > threshold:
            filtered_text_data.append(text_item)
            filtered_video_data.append(video_item)
            
            if problem_type == "free-form":
                filtered_text_freeform.append(text_item)
                filtered_video_freeform.append(video_item)
                freeform_count += 1
            elif problem_type == "multiple choice":
                filtered_text_multiple.append(text_item)
                filtered_video_multiple.append(video_item)
                multiple_count += 1
    
    # print(f"\n[Filter] Saving results to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    save_json(filtered_text_data, os.path.join(output_dir, "filtered_text.json"))
    save_json(filtered_video_data, os.path.join(output_dir, "filtered_video.json"))
    save_json(filtered_text_freeform, os.path.join(output_dir, "filtered_text_freeform.json"))
    save_json(filtered_video_freeform, os.path.join(output_dir, "filtered_video_freeform.json"))
    save_json(filtered_text_multiple, os.path.join(output_dir, "filtered_text_multiple.json"))
    save_json(filtered_video_multiple, os.path.join(output_dir, "filtered_video_multiple.json"))
    
    print(f"[Filter] Results:")
    print(f"  Total filtered: {len(filtered_text_data)}")
    print(f"  Free-form: {freeform_count}")
    print(f"  Multiple choice: {multiple_count}")
    print(f"  Not found: {not_found_count}")
    print("[Filter] Done!")
    
    return filtered_text_data, filtered_video_data


def select_by_problem_id(source_file, filtered_file, output_file):
    # print(f"[Select] Loading filtered file: {filtered_file}")
    
    if not Path(filtered_file).exists():
        print(f"Error: Filtered file not found: {filtered_file}")
        sys.exit(1)
    
    with open(filtered_file, 'r', encoding='utf-8') as f:
        filtered_data = json.load(f)
    
    problem_ids = set()
    if isinstance(filtered_data, list):
        for item in filtered_data:
            if 'problem_id' in item:
                problem_ids.add(item['problem_id'])
    elif isinstance(filtered_data, dict):
        if 'problem_id' in filtered_data:
            problem_ids.add(filtered_data['problem_id'])
    else:
        print("Error: Invalid filtered data format")
        sys.exit(1)
    
    # print(f"[Select] Found {len(problem_ids)} problem_ids in filtered file")
    
    # print(f"[Select] Loading source file: {source_file}")
    
    if not Path(source_file).exists():
        print(f"Error: Source file not found: {source_file}")
        sys.exit(1)
    
    with open(source_file, 'r', encoding='utf-8') as f:
        source_data = json.load(f)
    
    selected_data = []
    for item in source_data:
        if item.get('problem_id') in problem_ids:
            selected_data.append(item)
    
    # print(f"[Select] Saving {len(selected_data)} selected entries to {output_file}")
    save_json(selected_data, output_file)
    
    print(f"[Select] Selected: {len(selected_data)}/{len(problem_ids)} entries")
    
    if len(problem_ids) != len(selected_data):
        matched_ids = {item.get('problem_id') for item in selected_data}
        missing_ids = problem_ids - matched_ids
        
        if missing_ids:
            print(f"  ⚠ Not found ({len(missing_ids)} problem_ids): {sorted(list(missing_ids))[:10]}")
    # else:
    #     print("  ✓ All problem_ids matched successfully")
    
    print("[Select] Done!")


def print_usage():
    print("""
Unified data processing tool for reward-based dataset filtering.

Usage:
    python data_processor.py calculate <input_file> [output_file]
    python data_processor.py filter <text_file> <video_file> <output_dir> [threshold]
    python data_processor.py select <source_file> <filtered_file> <output_file>

Commands:
    calculate - Calculate and add mean_total_reward to datasets
    filter    - Filter data based on reward differences between text and video modalities
    select    - Select data by problem_id matching

Examples:
    python data_processor.py calculate input.json output.json
    python data_processor.py filter text_12k.json video_12k.json ./output 0.12
    python data_processor.py select new_12k.json filtered_text.json selected.json
""")


def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "calculate":
        if len(sys.argv) < 3:
            print("Error: Missing input file")
            print("Usage: python data_processor.py calculate <input_file> [output_file]")
            sys.exit(1)
        
        input_file = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) > 3 else None
        calculate_rewards(input_file, output_file)
    
    elif command == "filter":
        if len(sys.argv) < 5:
            print("Error: Missing required arguments")
            print("Usage: python data_processor.py filter <text_file> <video_file> <output_dir> [threshold]")
            sys.exit(1)
        
        text_file = sys.argv[2]
        video_file = sys.argv[3]
        output_dir = sys.argv[4]
        threshold = float(sys.argv[5]) if len(sys.argv) > 5 else 0.12
        
        filter_by_reward_diff(text_file, video_file, output_dir, threshold)
    
    elif command == "select":
        if len(sys.argv) < 4:
            print("Error: Missing required arguments")
            print("Usage: python data_processor.py select <source_file> <filtered_file> <output_file>")
            sys.exit(1)
        
        source_file = sys.argv[2]
        filtered_file = sys.argv[3]
        output_file = sys.argv[4] if len(sys.argv) > 4 else "selected_data.json"
        
        select_by_problem_id(source_file, filtered_file, output_file)
    
    else:
        print(f"Error: Unknown command '{command}'")
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()