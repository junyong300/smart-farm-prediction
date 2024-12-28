## Overview

Smart Farm Prediction combines environmental sensing with advanced machine learning to:
- Monitor farm weather conditions in real-time
- Predict future environmental conditions
- Forecast crop growth patterns
- Detect and predict pest issues
- Analyze internal and external environmental factors

## System Architecture

### Models
The system includes several specialized prediction models:

1. Environmental Models:
   - `internal_basic`: Predicts internal environment conditions
   - `internal_poc`: Proof of concept for internal predictions
   - `external_basic`: Basic external environment predictions
   - `external_wfs`: Weather forecast system integration
   - `internal_self`: Self-learning internal environment model

2. Growth Models:
   - `growth_basic`: Basic crop growth predictions
   - `growth_simple`: Simplified growth predictions
   - `pest_basic`: Pest detection and prediction

3. Data Processing:
   - AI Hub integration for environmental data
   - Pest observation processing
   - CSV and Excel file handling

## Prerequisites

* Python 3.8 or higher - [Download & install](https://www.python.org/downloads/)
* Poetry package manager - [Download & install](https://python-poetry.org)
* MySQL database
* GPU support for TensorFlow (recommended)

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd smart-farm-prediction
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Configure the database:
   - Create a MySQL database
   - Update database configuration in `config/common.conf`

## Configuration

The system uses a configuration file located at `config/common.conf`. Key configurations include:

- Database settings:
  - DB_TYPE
  - DB_HOST
  - DB_PORT
  - DB_USER
  - DB_PASSWORD
  - DB_DATABASE

```

### Acknowledgement
This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No. 2021-0-00907, Development of Adaptive and Lightweight Edge-Collaborative Analysis Technology for Enabling Proactively Immediate Response and Rapid Learning).
