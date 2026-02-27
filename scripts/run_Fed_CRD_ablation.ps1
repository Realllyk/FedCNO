param(
    [string]$PythonExe = "python",
    [string]$Device = "cuda:0",
    [string]$Vul = "reentrancy",
    [string]$NoiseType = "non_noise",
    [double]$NoiseRate = 0.3,
    [string]$ModelType = "",
    [string]$ModelTypes = "CBGRU,CGE",
    [int]$Epoch = 30,
    [int]$WarmUpEpoch = 25,
    [int]$Batch = 8,
    [int]$ClientNum = 4,
    [int]$NumNeigh = 5,
    [string]$Profile = "core4",
    [string]$CustomModes = "",
    [string]$Seeds = "1,2,3",
    [string]$LabName = "FedCRD_Ablation",
    [string]$ExpTag = "crd_abl",
    [switch]$RandomNoise,
    [switch]$Diff,
    [switch]$UseEmaAnchor,
    [string]$NoAmbVariant = "soft",
    [double]$ConstQ = 1.0
)

$ErrorActionPreference = "Stop"

function Get-ModesFromProfile {
    param(
        [string]$ProfileName,
        [string]$CustomModeStr
    )
    switch ($ProfileName.ToLower()) {
        "core4" { return @("full", "no_amb", "no_cal", "no_clip") }
        "core5" { return @("full", "no_amb", "no_amb_const", "no_cal", "no_clip") }
        "custom" {
            if ([string]::IsNullOrWhiteSpace($CustomModeStr)) {
                throw "Profile=custom requires -CustomModes, e.g. full,no_amb,no_cal,no_clip"
            }
            return $CustomModeStr.Split(",") | ForEach-Object { $_.Trim().ToLower() } | Where-Object { $_ -ne "" }
        }
        default { throw "Unsupported profile: $ProfileName. Use core4/core5/custom." }
    }
}

function Get-SeedList {
    param([string]$SeedStr)
    $vals = $SeedStr.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }
    if ($vals.Count -eq 0) {
        throw "Seeds cannot be empty."
    }
    return $vals
}

function Get-ModelTypeList {
    param(
        [string]$SingleModelType,
        [string]$ModelTypeStr
    )
    if (-not [string]::IsNullOrWhiteSpace($SingleModelType)) {
        return @($SingleModelType.Trim())
    }
    $vals = $ModelTypeStr.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }
    if ($vals.Count -eq 0) {
        throw "ModelTypes cannot be empty."
    }
    return $vals
}

$modes = Get-ModesFromProfile -ProfileName $Profile -CustomModeStr $CustomModes
$seedList = Get-SeedList -SeedStr $Seeds
$modelTypeList = Get-ModelTypeList -SingleModelType $ModelType -ModelTypeStr $ModelTypes

Write-Host "FedCRD ablation sweep start"
Write-Host "Profile=$Profile | Modes=$($modes -join ',') | Seeds=$($seedList -join ',')"
Write-Host "Task: vul=$Vul noise=$NoiseType($NoiseRate) models=$($modelTypeList -join ',') device=$Device"

foreach ($currentModelType in $modelTypeList) {
    foreach ($mode in $modes) {
        $modeArg = $mode
        $variantArg = $NoAmbVariant
        if ($mode -eq "no_amb_const") {
            $modeArg = "no_amb"
            $variantArg = "const"
        }

        foreach ($seed in $seedList) {
            $runTag = "${ExpTag}_${currentModelType}_${mode}_seed${seed}"
            $cmd = @(
                $PythonExe, "fed_main/Fed_CRD_ablation.py",
                "--device", $Device,
                "--vul", $Vul,
                "--noise_type", $NoiseType,
                "--noise_rate", "$NoiseRate",
                "--model_type", $currentModelType,
                "--epoch", "$Epoch",
                "--warm_up_epoch", "$WarmUpEpoch",
                "--batch", "$Batch",
                "--client_num", "$ClientNum",
                "--num_neigh", "$NumNeigh",
                "--lab_name", $LabName,
                "--seed", "$seed",
                "--crd_ablation_mode", $modeArg,
                "--crd_noamb_variant", $variantArg,
                "--crd_const_q", "$ConstQ",
                "--exp_tag", $runTag
            )

            if ($RandomNoise) { $cmd += "--random_noise" }
            if ($Diff) { $cmd += "--diff" }
            if ($UseEmaAnchor) { $cmd += @("--crd_q_anchor", "ema") }

            Write-Host "Running model=$currentModelType mode=$mode seed=$seed ..."
            & $cmd[0] $cmd[1..($cmd.Count - 1)]
            if ($LASTEXITCODE -ne 0) {
                throw "Run failed: model=$currentModelType mode=$mode seed=$seed"
            }
        }
    }
}

Write-Host "FedCRD ablation sweep completed."
