import uvicorn
import os


if __name__ == "__main__":
    print(os.curdir)
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)