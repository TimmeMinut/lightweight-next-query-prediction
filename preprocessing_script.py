import os
import re
import argparse
import datetime
import time

def preprocess_aol_query_log(input_dir):

    start_time = time.time()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    # Regular expression to find queries consisting of only special characters
    special_chars_only_pattern = re.compile(r'^[_\W\s]*$')

    total_processed_lines = 0
    total_duplicates_removed = 0
    total_special_chars_removed = 0
    total_malformed_ids_removed = 0
    total_malformed_lines_skipped = 0
    total_empty_lines_skipped = 0
    total_files = 0

    # Create output directory and sub-directory for processed files
    output_dir = f"aol_processed_{timestamp}"
    processed_dir = os.path.join(output_dir, "processed_files")
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    # Create filepaths for output files
    special_char_output_file = os.path.join(
        output_dir, "special_char_queries.txt")
    malformed_id_output_file = os.path.join(
        output_dir, "malformed_id_queries.txt")
    stats_output_file = os.path.join(output_dir, "processing_stats.txt")

    # Open miscellaneous output files for writing
    with open(special_char_output_file, 'w', encoding='utf-8') as special_char_file, \
            open(malformed_id_output_file, 'w', encoding='utf-8') as malformed_id_file, \
            open(stats_output_file, 'w', encoding='utf-8') as stats_file:

        print(f"AOL Query Log Processing - Started at {timestamp}\n")
        stats_file.write(
            f"AOL Query Log Processing - Started at {timestamp}\n")

        for filename in os.listdir(input_dir):
            if not filename.endswith('.txt'):
                continue

            total_files += 1

            # Create pathnames for input file and output file
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(processed_dir, filename)

            print(f"Processing file {filename}")
            stats_file.write(f"Processing file {filename}\n")

            file_processed_lines = 0
            file_duplicates_removed = 0
            file_special_chars_removed = 0
            file_malformed_ids_removed = 0
            file_malformed_lines_skipped = 0
            file_empty_lines_skipped = 0

            # Open input file and create output file
            try:
                with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:

                    next(infile)  # Skip the header

                    prev_anon_id = None
                    prev_query = None

                    for line in infile:
                        file_processed_lines += 1

                        line = line.strip()
                        if not line:
                            file_empty_lines_skipped += 1
                            continue  # Skip empty lines

                        parts = line.split('\t')
                        if len(parts) < 3:
                            file_malformed_lines_skipped += 1
                            continue  # Skip malformed lines

                        anon_id = parts[0].strip()
                        query = parts[1].strip()

                        if not anon_id.isdigit():
                            file_malformed_ids_removed += 1
                            malformed_id_file.write(
                                f"{line}\t{filename}\n")
                            # Skip malformed anonIDs
                            continue

                        is_duplicate = (
                            anon_id == prev_anon_id and query == prev_query)

                        is_special_chars_only = bool(
                            special_chars_only_pattern.match(query))

                        if is_duplicate:
                            file_duplicates_removed += 1
                        elif is_special_chars_only:
                            file_special_chars_removed += 1
                            special_char_file.write(line + '\n')
                        else:
                            # Only keep a query if its unique and not only consisting of special characters.
                            outfile.write(line + '\n')

                        prev_anon_id = anon_id
                        prev_query = query

                    total_processed_lines += file_processed_lines
                    total_duplicates_removed += file_duplicates_removed
                    total_special_chars_removed += file_special_chars_removed
                    total_malformed_ids_removed += file_malformed_ids_removed
                    total_malformed_lines_skipped += file_malformed_lines_skipped
                    total_empty_lines_skipped += file_empty_lines_skipped

                    file_stats = [
                        f"  - Processed: {file_processed_lines:,} queries",
                        f"  - Skipped {file_empty_lines_skipped:,} empty lines",
                        f"  - Skipped {file_malformed_lines_skipped:,} malformed lines",
                        f"  - Removed {file_malformed_ids_removed:,} queries with malformed IDs",
                        f"  - Removed {file_duplicates_removed:,} duplicate queries",
                        f"  - Removed {file_special_chars_removed:,} special-character-only queries",
                        f"  - Remaining: {file_processed_lines - file_duplicates_removed - file_special_chars_removed - file_malformed_ids_removed:,} queries\n"
                    ]

                    # Print and write the file stats
                    for stat in file_stats:
                        print(stat)
                        stats_file.write(stat + "\n")

            except Exception as e:
                error_msg = f"Error processing {filename}: {str(e)}"
                print(error_msg)
                stats_file.write(error_msg + "\n")

        remaining = total_processed_lines - total_duplicates_removed - \
            total_special_chars_removed - total_malformed_ids_removed

        summary_stats = [
            "\n" + "="*50,
            "PROCESSING COMPLETE",
            "="*50,
            f"Processed {total_files} files with {total_processed_lines:,} total queries",
            f"Skipped {total_empty_lines_skipped:,} empty lines",
            f"Skipped {total_malformed_lines_skipped:,} malformed lines",
            f"Removed {total_malformed_ids_removed:,} queries with malformed IDs",
            f"Removed {total_duplicates_removed:,} duplicate queries ({total_duplicates_removed/total_processed_lines*100:.2f}%)",
            f"Removed {total_special_chars_removed:,} special-char queries ({total_special_chars_removed/total_processed_lines*100:.2f}%)",
            f"Remaining non-duplicate, valid ID queries: {total_processed_lines - total_duplicates_removed - total_malformed_ids_removed:,} ({(total_processed_lines - total_duplicates_removed - total_malformed_ids_removed)/total_processed_lines*100:.2f}%)",
            f"Remaining non-duplicate, valid ID, non-special-char queries: {remaining:,} ({remaining/total_processed_lines*100:.2f}%)",
            "="*50,
        ]

        for stat in summary_stats:
            print(stat)
            stats_file.write(stat + "\n")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nProcessing completed at {datetime.datetime.now()}\n")
        print(f"Elapsed time: {elapsed_time} seconds")
        stats_file.write(
            f"\nProcessing completed at {datetime.datetime.now()}\n")
        stats_file.write(f"Elapsed time: {elapsed_time} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess AOL query log files")

    parser.add_argument(
        "input_dir", help="Path to input directory containing AOL query log files")

    args = parser.parse_args()

    preprocess_aol_query_log(args.input_dir)
