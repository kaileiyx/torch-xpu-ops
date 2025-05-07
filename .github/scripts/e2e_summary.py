import os
import subprocess
import sys
import glob
import shutil
import zipfile
import json
import re

def run_command(command):
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"Command failed: {command}\n{result.stderr}")
    return result.stdout

def get_model_result(results_dir):
    print("\n<table><thead>"
          "<tr>"
          "<th rowspan=2> Suite </th><th rowspan=2> Model </th>"
          "<th colspan=5> Training </th><th colspan=5> Inference </th>"
          "</tr><tr>"
          "<th> float32 </th><th> bfloat16 </th><th> float16 </th><th> amp_bf16 </th><th> amp_fp16 </th>"
          "<th> float32 </th><th> bfloat16 </th><th> float16 </th><th> amp_bf16 </th><th> amp_fp16 </th>"
          "</tr>"
          "</thead><tbody>")
    
    suite_list = sorted(set(
        os.path.basename(f).split('_')[1]
        for f in glob.glob(os.path.join(results_dir, "*_xpu_accuracy.csv"))
    ))

    def get_context(results_dir, suite, dtype, mode, model, colorful):
        csv_files = glob.glob(os.path.join(results_dir, f'*{suite}_{dtype}_{mode}_xpu_accuracy.csv'))   
        context_list = []

        for csv_file in csv_files:
            with open(csv_file, 'r') as file:
                lines = file.readlines()
            for line in lines:
                if f',{model},' in line:
                    columns = line.split(',')
                    if len(columns) > 3:
                        context_list.append(columns[3].strip())
        color = colorful.split()[0]
        if color == "black":
            context = ' '.join(context_list)
        else:
            context = ' '.join([f"\\$\\${{__color__{{{color}}}{value}}}\\$\\$" for value in context_list])
        
        return context
        
    for suite in suite_list:
        model_list = sorted(set(
            line.split(',')[1]
            for f in glob.glob(os.path.join(results_dir, f"*{suite}*_xpu_accuracy.csv"))
            for line in open(f) if line.startswith("xpu,")
        ))
        
        for model in model_list:
            for dtype in ["float32", "bfloat16", "float16", "amp_bf16", "amp_fp16"]:
                for mode in ["training", "inference"]:
                    colorful = run_command(f"grep -w {model} /tmp/tmp-{suite}-{mode}-{dtype}.txt 2>&1 | awk 'BEGIN{{color = \"black\"; exit_label = 0;}}"
                                           "{{if ($0 ~ /Real failed/){{color=\"red\"; exit_label++;}} else if ($0 ~ /Expected failed/){{color=\"blue\";}}"
                                           "else if ($0 ~ /Warning timeout/){{color=\"orange\";}} else if ($0 ~ /New models/){{color=\"blue\";}}"
                                           "else if ($0 ~ /Failed to passed/){{color=\"green\"; exit_label++;}}}} END{{print color, exit_label}}'")
                    with open("/tmp/tmp-result.txt", "a") as f:
                        f.write(colorful)
                    
                    context = get_context(results_dir, suite, dtype, mode, model, colorful)
                    os.environ[f"{mode}_{dtype}"] = context.strip()
            
            print(f"<tr>"
                  f"<td>{suite}</td>"
                  f"<td>{model}</td>"
                  f"<td>{os.getenv('training_float32', '')}</td>"
                  f"<td>{os.getenv('training_bfloat16', '')}</td>"
                  f"<td>{os.getenv('training_float16', '')}</td>"
                  f"<td>{os.getenv('training_amp_bf16', '')}</td>"
                  f"<td>{os.getenv('training_amp_fp16', '')}</td>"
                  f"<td>{os.getenv('inference_float32', '')}</td>"
                  f"<td>{os.getenv('inference_bfloat16', '')}</td>"
                  f"<td>{os.getenv('inference_float16', '')}</td>"
                  f"<td>{os.getenv('inference_amp_bf16', '')}</td>"
                  f"<td>{os.getenv('inference_amp_fp16', '')}</td>"
                  f"</tr>").replace('__color__', '\\color').replace('_', '\\_')
    
    print("</tbody></table>\n")

def main():
    results_dir = sys.argv[1]
    artifact_type = sys.argv[2]
    check_file = os.path.join(os.path.dirname(__file__), "../ci_expected_accuracy/check_expected.py")
    tmp_files = glob.glob("/tmp/tmp-*.txt")
    for tmp_file in tmp_files:
        os.remove(tmp_file)
    accuracy = len(glob.glob(os.path.join(results_dir,"**","*_xpu_accuracy.csv"),recursive=True))
    with open("/tmp/tmp-result.txt", "w") as f:
        f.write("")
    print (accuracy)
    if accuracy > 0:
        print("#### Note:\n"
              "$${\\color{red}Red}$$: the failed cases which need look into\n"
              "$${\\color{green}Green}$$: the new passed cases which need update reference\n"
              "$${\\color{blue}Blue}$$: the expected failed or new enabled cases\n"
              "$${\\color{orange}Orange}$$: the warning cases\n"
              "Empty means the cases NOT run\n\n")
        print("### Accuracy")
        print("| Category | Total | Passed | Pass Rate | $${\\color{red}Failed}$$ | "
              "$${\\color{blue}Xfailed}$$ | $${\\color{orange}Timeout}$$ | "
              "$${\\color{green}New Passed}$$ | $${\\color{blue}New Enabled}$$ | Not Run |")
        print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        
        with open("/tmp/tmp-summary.txt", "w") as f:
            f.write("")
        with open("/tmp/tmp-details.txt", "w") as f:
            f.write("")
        
        for csv in sorted(glob.glob(os.path.join(results_dir,"**", "*_xpu_accuracy.csv"),recursive=True)):
            category_match = re.search(r'inductor_(.*?)_xpu_accuracy', csv)
            category = category_match.group(1) if category_match else None
            suite_match = re.search(r'inductor_(.*?)_', csv)
            suite = suite_match.group(1) if suite_match else None
            if suite == 'timm':
                suite = 'timm_models'
            mode_match = re.search(r'_(.*?)_xpu_accuracy', csv)
            mode = mode_match.group(1).split('_')[-1] if mode_match else None
            dtype_match = re.search(r'inductor_[a-z]*_(.*?)_(infer|train)', csv)
            dtype = dtype_match.group(1).replace('models_', '') if dtype_match else None
            print (category)
            print (suite)
            print (mode)
            print (dtype)
            run_command(f"python {check_file} --suite {suite} --mode {mode} --dtype {dtype} --csv_file {csv} > /tmp/tmp-{suite}-{mode}-{dtype}.txt")
            
            with open(f"/tmp/tmp-{suite}-{mode}-{dtype}.txt", "r") as f:
                lines = f.readlines()
            
            total, passed, pass_rate, failed, xfail, timeout, new_passed, not_run, new_enabled = 0, 0, "", 0, 0, 0, 0, 0, 0
            for line in lines:
                if "Total" in line:
                    total = int(line.split()[2])
                elif "Passed" in line:
                    passed = int(line.split()[2])
                elif "Pass rate" in line:
                    pass_rate = line.split()[2]
                elif "Real failed" in line:
                    failed = int(line.split()[3])
                elif "Expected failed" in line:
                    xfail = int(line.split()[3])
                elif "timeout" in line:
                    timeout = int(line.split()[3])
                elif "Failed to passed" in line:
                    new_passed = int(line.split()[4])
                elif "Not run" in line:
                    not_run = int(line.split()[3])
                elif "New models" in line:
                    new_enabled = int(line.split()[2])
            
            with open("/tmp/tmp-summary.txt", "a") as f:
                f.write(f"| {category} | {total} | {passed} | {pass_rate} | {failed} | {xfail} | {timeout} | {new_passed} | {new_enabled} | {not_run} |\n")
        
        with open("/tmp/tmp-summary.txt", "r") as f:
            print(f.read())
        
        get_model_result(results_dir)
    
    performance = len(glob.glob(os.path.join(results_dir,"**", "*_xpu_performance.csv"),recursive=True))
    if performance > 0:
        print("### Performance")
        run_command("pip install jq > /dev/null 2>&1")
        if artifact_type:
            gh_api_command = (
                f"gh api --method GET -F per_page=100 -F page=10 "
                f"-H 'Accept: application/vnd.github+json' -H 'X-GitHub-Api-Version: 2022-11-28' "
                f"/repos/{os.getenv('GITHUB_REPOSITORY', 'intel/torch-xpu-ops')}/actions/artifacts "
                f"> {os.getenv('GITHUB_WORKSPACE', '/tmp')}/refer.json"
            )
            print (gh_api_command)
            run_command(gh_api_command)
            
        with open(f"{os.getenv('GITHUB_WORKSPACE', '/tmp')}/refer.json", "r") as f:
            artifacts = json.load(f)
            
            artifact_id = next(
                (artifact['id'] for artifact in artifacts['artifacts']
                 if artifact_type in artifact['name'] and artifact['workflow_run']['head_branch'] == "main"),
                None
            )
            print (artifact_id)
            if artifact_id:
                gh_api_command = (
                    f"gh api -H 'Accept: application/vnd.github+json' "
                    f"-H 'X-GitHub-Api-Version: 2022-11-28' "
                    f"/repos/{os.getenv('GITHUB_REPOSITORY', 'intel/torch-xpu-ops')}/actions/artifacts/{artifact_id}/zip "
                    f"> reference.zip"
                )
                run_command(gh_api_command)
        
        reference_dir = os.path.join(os.getenv('GITHUB_WORKSPACE', '/tmp'), 'reference')
        if os.path.exists(reference_dir):
            shutil.rmtree(reference_dir)
        os.makedirs(reference_dir)
        
        shutil.move("reference.zip", reference_dir)
        with zipfile.ZipFile(os.path.join(reference_dir, "reference.zip"), 'r') as zip_ref:
            zip_ref.extractall(reference_dir)
        
        #python_path = sys.executable
        run_command(f"python {os.path.join(os.path.dirname(__file__), 'perf_comparison.py')} -xpu {results_dir} -refer {reference_dir}")
        best_csv_path = os.path.join(results_dir, "best.csv")
        if os.path.exists(os.path.join(reference_dir, "best.csv")):
            shutil.copy(os.path.join(reference_dir, "best.csv"), best_csv_path)
        
        run_command(f"python {os.path.join(os.path.dirname(__file__), 'calculate_best_perf.py')} "
                    f"--new {results_dir} --best {best_csv_path} --device PVC1100 --os '{os.getenv('OS_PRETTY_NAME')}' "
                    f"--driver '{os.getenv('DRIVER_VERSION')}' --oneapi '{os.getenv('BUNDLE_VERSION')}' "
                    f"--gcc '{os.getenv('GCC_VERSION')}' --python '3.10' "
                    f"--pytorch '{os.getenv('TORCH_BRANCH_ID')}/{os.getenv('TORCH_COMMIT_ID')}' "
                    f"--torch-xpu-ops '{os.getenv('TORCH_XPU_OPS_COMMIT', os.getenv('GITHUB_SHA'))}' > /dev/null 2>&1")

if __name__ == "__main__":
    main()

