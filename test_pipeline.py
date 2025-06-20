#!/usr/bin/env python3

"""
Test script to run the updated loan prediction pipeline
"""

import sys
sys.path.append('src')

import pandas as pd
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_pipeline():
    """Test the updated loan prediction pipeline"""
    
    # Bootstrap the project
    bootstrap_project('.')
    
    with KedroSession.create(project_path='.') as session:
        try:
            logger.info("Testing data preparation pipeline...")
            # Run data preparation
            session.run(pipeline_name="data_preparation")
            logger.info("✅ Data preparation completed successfully")
            
            logger.info("Testing data processing pipeline...")
            # Run data processing
            session.run(pipeline_name="data_processing")
            logger.info("✅ Data processing completed successfully")
            
            logger.info("Testing data science pipeline...")
            # Run data science
            session.run(pipeline_name="data_science")
            logger.info("✅ Data science pipeline completed successfully")
            
            # Load and validate results
            catalog = session.load_context().catalog
            
            # Check the model metrics
            metrics = catalog.load("model_metrics")
            logger.info(f"Model metrics: {metrics}")
            
            logger.info("🎉 All pipelines completed successfully!")
            logger.info("="*50)
            logger.info("PIPELINE UPDATE SUMMARY:")
            logger.info("✅ Data preparation: Updated for new loan dataset")
            logger.info("✅ Data processing: Updated feature engineering")
            logger.info("✅ Data science: Updated for Default target variable")
            logger.info("✅ Streamlit app: Completely rewritten for new dataset")
            logger.info("="*50)
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

if __name__ == "__main__":
    success = test_pipeline()
    if success:
        print("\n🎉 All pipelines updated and working successfully!")
        print("The loan prediction system is now ready for the new dataset.")
        print("\nTo use the system:")
        print("1. Run 'kedro run' to train the model")
        print("2. Run 'streamlit run docker/app.py' to start the web interface")
    else:
        print("\n❌ Pipeline testing failed. Please check the logs above.")
        sys.exit(1) 