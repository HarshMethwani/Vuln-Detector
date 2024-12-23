import os
import csv
import re
import logging
from slither import Slither
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

CONTRACTS_DIR = "smartbugs-wild/processed_contracts"
CHUNK_SIZE = 1000

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function to set Solidity version
def set_solc_version(contract_path):
    with open(contract_path, "r", encoding="utf-8") as file:
        content = file.read()

    match = re.search(r"pragma solidity\s+([^\s;]+);", content)
    if not match:
        logger.warning(f"No Solidity version specified in {contract_path}")
        return False

    version = match.group(1).replace("^", "").replace("=", "").replace(">", "").strip()
    try:
        subprocess.run(["solc-select", "use", version], check=True, stdout=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        logger.error(f"Failed to set Solidity version to {version} for {contract_path}")
        return False

# Function to process each contract row
def process_contract_row(row):
    contract_address = row.get("contract_address")
    function_name = row.get("function_name")
    vulnerability_type = row.get("vulnerability")
    sol_file = os.path.join(CONTRACTS_DIR, f"{contract_address}.sol")

    if not os.path.exists(sol_file):
        return {"contract_address": contract_address, "function_name": function_name, "vulnerability_type": vulnerability_type, "code_snippet": "File not found"}

    if not set_solc_version(sol_file):
        return {"contract_address": contract_address, "function_name": function_name, "vulnerability_type": vulnerability_type, "code_snippet": "Failed to set compatible solc version"}

    try:
        slither = Slither(sol_file)

        with open(sol_file, "r", encoding="utf-8") as file:
            source_code = file.read()

        for contract in slither.contracts:
            for function in contract.functions_entry_points:
                if function.name == function_name:
                    start = function.source_mapping.start
                    length = function.source_mapping.length
                    function_body = source_code[start:start + length]
                    return {
                        "contract_address": contract_address,
                        "function_name": function_name,
                        "vulnerability_type": vulnerability_type,
                        "code_snippet": function_body
                    }
        return {"contract_address": contract_address, "function_name": function_name, "vulnerability_type": vulnerability_type, "code_snippet": "Function not found"}
    except Exception as e:
        return {"contract_address": contract_address, "function_name": function_name, "vulnerability_type": vulnerability_type, "code_snippet": f"Error: {str(e)}"}

# Function to process CSV file
def process_csv(input_csv, output_csv):
    with open(input_csv, "r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)

    results = []
    total_rows = len(rows)
    logger.info(f"Starting processing of {total_rows} rows.")

    with ProcessPoolExecutor() as executor:
        future_to_row = {executor.submit(process_contract_row, row): row for row in rows}

        for i, future in enumerate(as_completed(future_to_row), start=1):
            try:
                result = future.result()
                if result["code_snippet"] not in ["File not found", "Failed to set compatible solc version", "Function not found"] and not result["code_snippet"].startswith("Error"):
                    results.append(result)
            except Exception as e:
                logger.error(f"Error processing row: {future_to_row[future]}. Exception: {e}")

            if i % 100 == 0:
                logger.info(f"Processed {i} functions.")

            if i % CHUNK_SIZE == 0 or i == total_rows:
                save_results(output_csv, results)
                results.clear()
                logger.info(f"Saved {i}/{total_rows} rows to output.")

    logger.info("Processing complete.")

# Function to save results to CSV
def save_results(output_csv, results):
    if results:
        write_mode = "w" if not os.path.exists(output_csv) else "a"
        with open(output_csv, write_mode, newline="", encoding="utf-8") as outfile:
            fieldnames = ["contract_address", "function_name", "vulnerability_type", "code_snippet"]
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            if write_mode == "w":
                writer.writeheader()
            writer.writerows(results)

if __name__ == "__main__":
    INPUT_CSV = "vulnerabilities_slither_icse20_cleaned.csv"
    OUTPUT_CSV = "vulnerabilities_with_code.csv"
    process_csv(INPUT_CSV, OUTPUT_CSV)
