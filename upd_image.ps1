Write-Host "Building image..."
docker image build --file Dockerfile --tag dqnexploration:test1 ./
Write-Host "Exported."