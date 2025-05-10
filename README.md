
```markdown
# SynteticDataGenerator

A Streamlit application designed to generate synthetic search data related to the Bhagavad Gita. This tool allows users to customize search queries, sentiment categories, and various metrics to simulate realistic search data for analysis and testing purposes.

## Features

- **Customizable Queries and Sentiments**: Define your own search queries and sentiment categories.
- **Advanced Metrics Settings**: Adjust ranges for click-through rate, bounce rate, and session duration.
- **Interactive Analytics**: Visualize data with interactive charts and graphs.
- **Data Export**: Download generated data in CSV, JSON, or Excel formats.
- **Configuration Saving**: Save and load custom configurations for reuse.
- **Comprehensive Reports**: Generate detailed reports with key insights.

## Installation

To run this application locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/SyenteticDataGenerator.git
   cd Syentetic-Data-Generator
   ```

2. **Set Up a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Input Parameters**:
   - Enter search queries and sentiment categories.
   - Adjust the number of samples per query and the date range for timestamps.
   - Customize metric ranges in the advanced settings.

2. **Generate Data**:
   - Click the "Generate Synthetic Data" button to create the dataset.
   - Preview the data and proceed to generate the full dataset.

3. **View and Download**:
   - Explore the generated data in the "Generated Data" tab.
   - Use the "Interactive Analytics" tab to visualize data with charts and graphs.
   - Download the data in your preferred format (CSV, JSON, or Excel).

4. **Save Configurations**:
   - Save your current configuration for future use.
   - Load previously saved configurations to quickly generate data with the same settings.

## Requirements

- Python 3.7 or higher
- Streamlit
- Pandas
- Plotly
- XlsxWriter
- OpenPyXL

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, please contact [venkat.chavan.n@gmail.com].

