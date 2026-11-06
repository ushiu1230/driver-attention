param([Parameter(Mandatory=$true)][ValidateSet('E0','E1','E2')][string]$Experiment)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$runNames = @{ E0='E0-full-finetune'; E1='E1-partial-freeze'; E2='E2-discriminative-lr' }
$runName = $runNames[$Experiment]
$runDir = Join-Path 'checkpoints' $runName
if (Test-Path -LiteralPath $runDir) { throw "Run folder exists: $runDir. Do not overwrite an existing or active run." }
$init = 'checkpoints/tsm-attn-aug-fast-seed123/best.pt'
if (-not (Test-Path -LiteralPath $init)) { throw "Missing shared initialization: $init" }
$initHash = (Get-FileHash -LiteralPath $init -Algorithm SHA256).Hash
$sourceFiles = @('main.py','train.py','dataset/dataset.py','models/drivergaze_net.py','models/model_factory.py','models/saliency_decoder.py','utils/losses.py','utils/metrics.py')
$codeHashes = @{}
foreach ($file in $sourceFiles) { $codeHashes[$file] = (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash }
$csvHashes = @{}
foreach ($split in @('train','val')) {
    foreach ($csvFile in (Get-ChildItem -LiteralPath "data/raw/drivergaze360/trainval/dataset/$split" -Filter sim_gaze_df.csv -Recurse -File)) {
        $csvHashes[$csvFile.FullName] = (Get-FileHash -LiteralPath $csvFile.FullName -Algorithm SHA256).Hash
    }
}
New-Item -ItemType Directory -Path $runDir | Out-Null
$receipt = @{ run=$Experiment; initialization=$init; initialization_sha256=$initHash; source_sha256=$codeHashes; csv_sha256=$csvHashes; started=(Get-Date).ToString('o'); note='Source/CSV hashes recorded before foreground training; image bytes not hashed. No process controller or shutdown scheduled.' }
$receipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $runDir 'run_manifest.json') -Encoding utf8
$cli = @('--name',$runName,'--save-dir','checkpoints','--experiment','tsm_attn_pool',
 '--train-path','data/raw/drivergaze360/trainval/dataset/train',
 '--val-path','data/raw/drivergaze360/trainval/dataset/val',
 '--init-weights',$init,'--no-pretrained','--num-epochs','10','--early-stopping-patience','10',
 '--batch-size','1','--accumulation-steps','5','--lr','1e-6','--weight-decay','0.01',
 '--num-workers','2','--clip-step','8','--frame-stride','1','--skip-first','30',
 '--img-size','224','1120','--resize-mode','stretch','--seed','123','--dropout','0',
 '--use-amp','--paper-protocol','--loss-profile','drivergaze360','--w-kld','1','--w-cc','1','--w-mse','0','--w-nss','0')
if ($Experiment -eq 'E1') { $cli += @('--freeze-backbone-stages','12') }
if ($Experiment -eq 'E2') { $cli += @('--backbone-lr-scale','0.1') }
& python main.py @cli
$trainingExit = $LASTEXITCODE
$receipt['finished'] = (Get-Date).ToString('o')
$receipt['exit_code'] = $trainingExit
$receipt['initialization_unchanged'] = ((Get-FileHash -LiteralPath $init -Algorithm SHA256).Hash -eq $initHash)
$receipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $runDir 'run_manifest.json') -Encoding utf8
if (-not $receipt['initialization_unchanged']) { throw 'Initialization changed during training.' }
if ($trainingExit -ne 0) { throw "Training exited with code $trainingExit" }
