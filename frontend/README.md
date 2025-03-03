# MMT Frontend

[![React](https://img.shields.io/badge/React-18.2.0-61DAFB?logo=react)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.2.2-3178C6?logo=typescript)](https://www.typescriptlang.org/)
[![Material UI](https://img.shields.io/badge/Material_UI-5.14.5-0081CB?logo=material-ui)](https://mui.com/)
[![Recharts](https://img.shields.io/badge/Recharts-2.10.4-22B5BF)](https://recharts.org/)

This is the frontend application for the Model Monitoring Tool (MMT), built with React, TypeScript, and Material-UI. The frontend provides an intuitive user interface for monitoring and analyzing credit risk models, visualizing model performance metrics, and managing model data.

## 🚀 Features

### 💻 User Interface
- **Responsive Design**: Adapts to different screen sizes for optimal viewing experience
- **Theme Toggle**: Switch between light and dark themes based on user preference
- **Navigation Drawer**: Easy access to all application features
- **Breadcrumbs**: Clear indication of current location within the application
- **Error Boundaries**: Graceful handling of runtime errors

### 📈 Data Visualization
- **Interactive Charts**: Dynamic charts using Recharts library
- **Time Series Analysis**: Visualize trends over time
- **Distribution Charts**: Histograms and density plots for data distribution
- **Heat Maps**: Visualize segmentation analysis
- **WOE Plots**: Weight of Evidence visualization for variable importance

### 📊 Model Analysis
- **PD Model**: Comprehensive analysis of Probability of Default models
- **Macro Model**: Statistical tests and visualizations for macroeconomic models
- **LGD Model**: Loss Given Default analysis with segmentation
- **EAD Model**: Exposure at Default analysis with CCF distribution
- **Summary Dashboard**: Overview of key metrics across all models

### 💾 Data Management
- **File Upload**: Upload model data files with validation
- **Database Manager**: View and manage uploaded datasets
- **User Thresholds**: Configure model thresholds and criteria

## 🛠️ Tech Stack

- **React 18**: Modern UI library with hooks and functional components
- **TypeScript**: Type-safe JavaScript for improved developer experience
- **Material-UI v5**: Component library for consistent design
- **React Router v6**: Navigation and routing with modern API
- **Axios**: Promise-based HTTP client for API calls
- **Recharts**: Composable charting library built on React components
- **Notistack**: Snackbar notifications for user feedback
- **Vite**: Next-generation frontend tooling for faster development

## 📍 Project Structure

```
src/
├── components/         # Reusable UI components
│   ├── Layout.tsx     # Main application layout
│   ├── ThemeToggle.tsx # Dark/light theme toggle
│   └── ...
├── pages/             # Main application pages
│   ├── PDModel.tsx    # PD model analysis page
│   ├── MacroModel.tsx # Macro model analysis page
│   ├── LGDModel.tsx   # LGD model analysis page
│   ├── EADModel.tsx   # EAD model analysis page
│   ├── Summary.tsx    # Summary dashboard page
│   ├── DataUpload.tsx # Data upload page
│   └── ...
├── utils/             # Utility functions
│   ├── api.ts         # API client for backend communication
│   ├── theme.ts       # Theme configuration
│   └── ...
├── contexts/          # React context providers
│   ├── ThemeContext.tsx # Theme context provider
│   └── ...
├── types/             # TypeScript type definitions
│   └── ...
├── App.tsx            # Main application component
├── main.tsx           # Application entry point
└── vite-env.d.ts      # Vite environment types
```

## 🔧 Key Components

### Layout Component

The `Layout` component provides the main application structure with a navigation drawer, app bar, and content area. It includes the theme toggle and handles responsive layout adjustments.

### Data Visualization Components

The application includes several reusable chart components built with Recharts:

- **TimeSeriesChart**: Visualize time series data with customizable options
- **DistributionChart**: Display data distributions with histograms
- **WOEPlot**: Weight of Evidence visualization for variable importance

### Model Analysis Pages

Each model type has a dedicated analysis page with specialized components:

- **PDModel**: Probability of Default model analysis with discrimination, calibration, and stability metrics
- **MacroModel**: Macroeconomic model analysis with stationarity tests and performance metrics
- **LGDModel**: Loss Given Default model analysis with recovery rate analysis
- **EADModel**: Exposure at Default model analysis with CCF distribution

## 💻 Development

### Prerequisites

- Node.js 16 or higher
- npm 8 or higher

### Installation

1. Clone the repository and navigate to the frontend directory:
   ```bash
   git clone https://github.com/yourusername/mmt.git
   cd mmt/frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```
   The application will be available at http://localhost:5173

### Available Scripts

- `npm run dev`: Start the development server
- `npm run build`: Build the application for production
- `npm run lint`: Lint the codebase
- `npm run preview`: Preview the production build locally

## 🚀 Production Build

1. Create a production build:
   ```bash
   npm run build
   ```

2. The build artifacts will be stored in the `dist/` directory, ready to be deployed to a static hosting service or served by a web server.

## 🔗 API Integration

The frontend communicates with the backend API running on `http://localhost:5000` by default. The API client is defined in `src/utils/api.ts` and provides methods for all backend interactions.

### API Client

The API client uses Axios for HTTP requests and includes error handling and response transformation. Key methods include:

- **Data Upload**: `uploadPDData()`, `uploadLGDData()`, `uploadEADData()`, `uploadMacroData()`
- **Analysis**: `analyzePD()`, `analyzeLGD()`, `analyzeEAD()`, `analyzeMacro()`
- **Configuration**: `getThresholds()`, `saveThresholds()`, `getSummary()`
- **Data Management**: `getDatasets()`, `deleteDataset()`, `getAnalysisResults()`

## 🎨 Theme Customization

The application supports both light and dark themes using Material-UI's theming system. The theme preference is stored in local storage for persistence across sessions.

### Theme Toggle

The `ThemeToggle` component allows users to switch between light and dark themes. The theme context manages the current theme state and provides it to all components.

### Custom Theme

The theme is defined in `src/utils/theme.ts` and includes customizations for:

- Color palette (primary, secondary, error, warning, info, success)
- Typography (font family, sizes, weights)
- Component styling (buttons, cards, tables, etc.)
- Dark mode adjustments (background colors, text colors, etc.)

## 🔍 Accessibility

The application follows accessibility best practices:

- Proper semantic HTML elements
- ARIA attributes where needed
- Keyboard navigation support
- Color contrast compliance
- Screen reader compatibility

## 🔎 Browser Compatibility

The application is compatible with modern browsers:

- Chrome (latest 2 versions)
- Firefox (latest 2 versions)
- Safari (latest 2 versions)
- Edge (latest 2 versions)

## 📝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
