# Publish Commands: Monitoring Data Warehouse

Create an empty GitHub repository named `monitoring-data-warehouse`, then run:

```powershell
Set-Location d:\GitHub\standalone-repos\monitoring-data-warehouse
git init
git add .
git commit -m "Initial standalone release"
git branch -M main
git remote add origin https://github.com/matthew-lottly/monitoring-data-warehouse.git
git push -u origin main
```

If the repository already has a remote configured, skip the `git remote add origin ...` line.