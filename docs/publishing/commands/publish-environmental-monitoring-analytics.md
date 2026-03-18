# Publish Commands: Environmental Monitoring Analytics

Create an empty GitHub repository named `environmental-monitoring-analytics`, then run:

```powershell
Set-Location d:\GitHub\standalone-repos\environmental-monitoring-analytics
git init
git add .
git commit -m "Initial standalone release"
git branch -M main
git remote add origin https://github.com/matthew-lottly/environmental-monitoring-analytics.git
git push -u origin main
```

If the repository already has a remote configured, skip the `git remote add origin ...` line.