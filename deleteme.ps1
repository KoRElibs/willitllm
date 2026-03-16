$json = '{"model":"gemma1.3"}'; $res = Invoke-WebRequest -Uri "http://localhost:11434/api/show" -Method POST -Body $json -ContentType "application/json"
$res.Content | ConvertFrom-Json