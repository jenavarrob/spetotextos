import secrets

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials


security = HTTPBasic()


def authenticate(
    request: Request,
    credentials: HTTPBasicCredentials = Depends(security),
) -> str:
    settings = request.app.state.settings
    valid_user = secrets.compare_digest(credentials.username, settings.app_username)
    valid_password = secrets.compare_digest(credentials.password, settings.app_password)
    if not (valid_user and valid_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

