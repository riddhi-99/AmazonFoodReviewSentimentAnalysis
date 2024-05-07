try{
    python -m pip install wordcloud
    python -m pip install STOPWORDS
    python -m pip install Review
    python -m pip install tqdm
    $Path = (Get-Location).Path
    python $Path\read-text-data.py
}
catch{
    Write-Host $_
}

Start-Sleep -Seconds 60