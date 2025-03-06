# StockCaster Clarifier

## Overview

StockCaster Clarifier is an advanced web-based platform for visualizing and analyzing stock market data. It provides interactive charts with customizable technical indicators, allowing users to gain insights from time series financial data.

## Features

- **Interactive Stock Charts**: View candlestick charts with adjustable time frames
- **Technical Indicators**: Toggle various indicators on/off (SMA, EMA, Bollinger Bands, etc.)
- **CSV Data Import**: Load your own stock data or use the provided sample data
- **Time Frame Selection**: Easily switch between different time periods
- **Data Statistics**: View key statistics about the selected data range
- **Responsive Design**: Works on both desktop and mobile devices

## Technologies Used

- **Next.js**: React framework for building the web application
- **TypeScript**: For type-safe code
- **Recharts**: For rendering interactive charts
- **Tailwind CSS**: For styling the user interface
- **Papa Parse**: For parsing CSV data files

## Getting Started

### Prerequisites

- Node.js 14.x or higher
- npm or yarn package manager

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd stockcaster-clarifier
   ```

2. Install dependencies:
   ```
   npm install
   # or
   yarn install
   ```

3. Run the development server:
   ```
   npm run dev
   # or
   yarn dev
   ```

4. Open [http://localhost:3000](http://localhost:3000) in your browser to see the application.

## Usage

### Loading Data

StockCaster Clarifier can load stock data from CSV files. The application expects CSV files with at least the following columns:
- `datetime`: The timestamp for each data point
- `open`, `high`, `low`, `close`: Price data
- Additional technical indicators (optional)

You can either:
- Use the "Load Sample Data" button to load the provided SPY data
- Upload your own CSV file using the file uploader

### Customizing the Chart

- Use the time frame selector to change the date range
- Toggle technical indicators on/off from the Indicators panel
- Hover over the chart to see detailed values at specific points

## Project Structure

```
stockcaster-clarifier/
├── public/
│   └── data/           # Sample data files
├── src/
│   ├── app/            # Next.js app router
│   ├── components/     # React components
│   │   ├── charts/     # Chart components
│   │   ├── layout/     # Layout components
│   │   └── ui/         # UI components
│   ├── lib/            # Utility functions and context
│   ├── styles/         # Global styles
│   └── types/          # TypeScript type definitions
└── ...configuration files
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 