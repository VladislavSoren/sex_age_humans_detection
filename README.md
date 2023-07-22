# Service for detecting human faces and determining gender and age in an image.

## Service functionality:
- Human Face Detection: Dual Shot Face Detector (DSFD detector)
- Human Gender Determination: SSR-Net (Soft Stagewise Regression Network)
- Determination of human age: SSR-Net

## Architecture:
- Input image processing module
- DSFD detector - finds fragments with faces in the image
- SSR-Net - determines the gender of people in found fragments
- Second SSR-Net - determines the age of people in found fragments
- Module for applying predictions to the original image

## Project implementation:
- Implemented API based on FastAPI
- Production server - uvicorn
- Service running in Docker

## API Description:
- Input json format is controlled (pydantic)
- *Functionality will be expanded