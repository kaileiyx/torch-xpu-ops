import os
import re
import sys
import subprocess
import pandas as pd
from pathlib import Path
import glob
import zipfile

def run_command(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout.strip()

def get_model_result(results_dir):
    suite_list = sorted(set(
        os.path.basename(f).split('_')[1]
        for f in glob.glob(os.path.join(results_dir, "**", "*_xpu_accuracy.csv"))
    ))
    table = """
    <table><thead>
        <tr>
            <th rowspan=2> Suite </th><th rowspan=2> Model </th>
            <th colspan=5> Training </th><th colspan=5> Inference </th>
        </tr><tr>
            <th> float32 </th><th> bfloat16 </th><th> float16 </th><th> amp_bf16 </th><th> amp_fp16 </th>
            <th> float32 </th><th> bfloat16 </th><th> float16 </th><th> amp_bf16 </th><th> amp_fp16 </th>
        </tr>
    </thead><tbody>
    """
    for suite in suite_list:
        model_list = sorted(set(
            pd.read_csv(f).query("xpu == 'xpu'")['model'].unique()
            for f in glob.glob(os.path.join(results_dir, "**", f"*{suite}*_xpu_accuracy.csv"))
        ))
        for model in model_list:
            for dtype in ["float32", "bfloat16", "float16", "amp_bf16", "amp_fp16"]:
                for mode in ["training", "inference"]:
                    colorful = run_command(f"grep -w {model} /tmp/tmp-{suite}-{mode}-{dtype}.txt")
                    context = run_command(f"grep {model} {suite} | cut -d, -f4")
                    context = context if "black" in colorful else f"\\$\\${{__color__{{{colorful.split()[0]}}}}}{context}\\$\\$"
                    exec(f"{mode}_{dtype} = '{context}'")
            table += f"""
            <tr>
                <td>{suite}</td>
                <td>{model}</td>
                <td>{training_float32}</td>
                <td>{training_bfloat16}</td>
                <td>{training_float16}</td>
                <td>{training_amp_bf16}</td>
                <td>{training_amp_fp16}</td>
                <td>{inference_float32}</td>
                <td>{inference_bfloat16}</td>
                <td>{inference_float16}</td>
                <td>{inference_amp_bf16}</td>
                <td>{inference_amp_fp16}</td>
            </tr>
            """
    table += "</tbody></table>\n"
    return table

def main(results_dir, reference_dir):
    check_file = os.path.join(os.path.dirname(__file__), "../ci_expected_accuracy/check_expected.py")
    compare_file = os.path.join(os.path.dirname(__file__), "perf_comparison.py")
    calculate_file = os.path.join(os.path.dirname(__file__), "calculate_best_perf.py")
    # Remove temporary files
    for tmp_file in glob.glob("/tmp/tmp-*.txt"):
        try:
            os.remove(tmp_file)
        except OSError as e:
            print(f"Error: {tmp_file} : {e.strerror}")

    accuracy = len(list(Path(results_dir).rglob("*_xpu_accuracy.csv")))
    if accuracy > 0:
        print("#### Note:")
        print(r"\$\${\\color{red}Red}\$\$: the failed cases which need look into")
        print(r"\$\${\\color{green}Green}\$\$: the new passed cases which need update reference")
        print(r"\$\${\\color{blue}Blue}\$\$: the expected failed or new enabled cases")
        print(r"\$\${\\color{orange}Orange}\$\$: the warning cases")
        print("Empty means the cases NOT run\n\n")
        print("### Accuracy")
        print(r"| Category | Total | Passed | Pass Rate | \$\${\\color{red}Failed}\$\$ | \$\${\\color{blue}Xfailed}\$\$ | \$\${\\color{orange}Timeout}\$\$ | \$\${\\color{green}New Passed}\$\$ | \$\${\\color{blue}New Enabled}\$\$ | Not Run |")
        print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")

        summary = []
        total = passed = failed = xfail = timeout = new_passed = new_enabled = not_run = 0
        pass_rate = failed_models = xfail_models = timeout_models = new_passed_models = not_run_models = new_enabled_models = ""
        csv_files = glob.glob(os.path.join(results_dir, "**", "*_xpu_accuracy.csv"), recursive=True)
        csvFiles = sorted(csv_files)
        
        for csv in csvFiles:
            basename = os.path.basename(csv)
            category_match = re.search(r'inductor_(.*?)_xpu_accuracy', basename)
            category = category_match.group(1) if category_match else ''
    
            suite_match = re.search(r'inductor_(.*?)_', basename)
            suite = suite_match.group(1).replace('timm', 'timm_models') if suite_match else ''
    
            mode_match = re.search(r'_(.*?)_xpu_accuracy', basename)
            mode = mode_match.group(1) if mode_match else ''
            mode_parts = mode.split('_')
            if 'training' in mode_parts:
                mode = 'training'
            elif 'inference' in mode_parts:
                mode = 'inference'
            else:
                mode = ''
    
            dtype_match = re.search(r'inductor_[a-z]*_(.*?)_(infer|train)', basename)
            dtype = dtype_match.group(1).replace('models_', '') if dtype_match else ''
            
            cmd = "python " + check_file + " --suite " + suite + " --mode " + mode + " --dtype " + dtype + " --csv_file " + csv + " > /tmp/tmp-" + suite + "-" + mode + "-" + dtype + ".txt"
            result = run_command(cmd)
            file_path = "/tmp/" + "tmp-" + suite + "-" + mode + "-" + dtype + ".txt"
            with open(file_path, 'r') as file:
                for line in file:
                    line = line.replace(', ', ',')
                    if "Total" in line:
                        total = int(line.split()[2])
                    elif "Passed" in line:
                        passed = int(line.split()[2])
                    elif "Pass rate" in line:
                        pass_rate = line.split()[2]
                    elif "Real failed" in line:
                        failed = int(line.split()[3])
                        failed_models = line.split()[4]
                    elif "Expected failed" in line:
                        xfail = int(line.split()[3])
                        xfail_models = line.split()[4]
                    elif "timeout" in line:
                        timeout = int(line.split()[3])
                        timeout_models = line.split()[4]
                    elif "Failed to passed" in line:
                        new_passed = int(line.split()[4])
                        new_passed_models = line.split()[5]
                    elif "Not run" in line:
                        not_run = int(line.split()[3])
                        not_run_models = line.split()[4]
                    elif "New models" in line:
                        new_enabled = int(line.split()[2])
                        new_enabled_models = line.split()[3]          
            summary.append(f"| {category} | {total} | {passed} | {pass_rate} | {failed} | {xfail} | {timeout} | {new_passed} | {new_enabled} | {not_run} |")
        
        for line in summary:
            print(line)    
        print(get_model_result(results_dir))

    performance = len(list(Path(results_dir).rglob("*_xpu_performance.csv")))
    if performance > 0:
        print("### Performance")
        # unzip reference zip package
        files = os.listdir(reference_dir)
        for file in files:
            if file.endswith('.zip'):
                zip_path = os.path.join(reference_dir, file)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(reference_dir)
                print("Extraction complete.")
        is_pr = os.getenv("IS_PR")
        if is_pr == "1":
            cmd_compare = "python " +  compare_file + " --xpu " + results_dir + " --refer " + reference_dir + " --pr"
            print (cmd_compare)
        else:
            cmd_compare = "python " +  compare_file + " --xpu " + results_dir + " --refer " + reference_dir
            print (cmd_compare)
        os.system(cmd_compare)
        best_csv_path = os.path.join(reference_dir, 'best.csv')
        if os.path.isfile(best_csv_path):
            cmd_copyBestcsv = "scp " + reference_dir + "/best.csv " + results_dir + "/best.csv"
        else:
            print("No best.csv file")
        # calculate_best_perf
        resultBestcsv = results_dir + "/best.csv"
        calculate_os = str(os.getenv('OS_PRETTY_NAME'))
        calculate_os_string = f'"{calculate_os}"'
        driver_version = str(os.getenv('DRIVER_VERSION'))
        oneapi_version = str(os.getenv('BUNDLE_VERSION'))
        gcc_version = str(os.getenv('GCC_VERSION'))
        python_version = str(os.getenv('python'))
        pytorch_branch = str(os.getenv('TORCH_BRANCH_ID'))
        pytorch_commitID = str(os.getenv('TORCH_COMMIT_ID'))
        xpu_ops = str(os.getenv('TORCH_XPU_OPS_COMMIT'))
        githubSHA = str(os.getenv('GITHUB_SHA'))
        
        if xpu_ops != "":
            cmd_calculate =r"python " + calculate_file + " --new " + results_dir + " --best " + resultBestcsv + " --device PVC1100 --os " + calculate_os_string + " --driver " + driver_version + " --oneapi " + oneapi_version + " --gcc " + gcc_version + " --python " + python_version + " --pytorch " + pytorch_branch + "/" + pytorch_commitID + "  --torch-xpu-ops " + xpu_ops
        else:
            cmd_calculate =r"python " + calculate_file + " --new " + results_dir + " --best " + resultBestcsv + " --device PVC1100 --os " + calculate_os_string + " --driver " + driver_version + " --oneapi " + oneapi_version + " --gcc " + gcc_version + " --python " + python_version + " --pytorch " + pytorch_branch + "/" + pytorch_commitID + "  --torch-xpu-ops " + githubSHA
        print (cmd_calculate)
        os.system(cmd_compare)
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <results_dir> <reference_dir>")
        sys.exit(1)
    results_dir = sys.argv[1]
    reference_dir = sys.argv[2]
    main(results_dir, reference_dir)
