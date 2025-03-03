import axios from 'axios';

// === Types ===
interface ApiError {
  response?: {
    data?: {
      error?: string;
      details?: string;
      message?: string;
    };
  };
  message: string;
}

interface StationarityTestResult {
  test_statistic: number;
  p_value: number;
  interpretation: string;
}

interface MacroAnalysisResult {
  stationarity_tests: Record<string, Record<string, StationarityTestResult>>;
  correlation_matrix: Record<string, Record<string, number>>;
  trend_analysis: Record<string, { mean: number; std: number; min: number; max: number; trend_slope: number }>;
}

interface RatingTransitionResult {
  unique_ratings: number;
  total_transitions: number;
  stability_rate: number;
  transition_matrix: Record<string, Record<string, number>>;
  significant_migrations: Record<string, Record<string, number>>;
}

interface AnalysisParams {
  quarter?: string;
  portfolio?: string;
  modelName?: string;
}

interface LGDUploadResponse {
  message?: string;
  error?: string;
  details?: string;
  filename?: string;
  data_preview?: any[];
}

interface LGDDropdownOptionsResponse {
  portfolios: string[];
  quarters: string[];
  modelNames: string[];
}

// === Constants ===
const BASE_URL = 'http://localhost:5000';

// === Utility Functions ===
const isAxiosError = (error: unknown): error is ApiError => (
  typeof error === 'object' &&
  error !== null &&
  'response' in error &&
  'message' in error
);

const handleApiError = (error: unknown, context: string): never => {
  console.error(`${context} Error:`, error);
  if (isAxiosError(error)) {
    const serverError = error.response?.data;
    const message = serverError?.error || serverError?.details || serverError?.message || error.message;
    throw new Error(`${context} Failed: ${message}`);
  }
  throw new Error(`${context} Failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
};

// === API Methods ===
export const api = {
  analyzePD: async (data?: any) => {
    try {
      const response = await axios.post(`${BASE_URL}/api/analyze/pd`, data || {}, {
        headers: { 'Content-Type': 'application/json' },
      });
      return response.data;
    } catch (error) {
      handleApiError(error, 'PD Analysis');
    }
  },

  analyzeMacroModel: async () => {
    try {
      const response = await axios.post(`${BASE_URL}/api/analyze/macro`, {}, {
        headers: { 'Content-Type': 'application/json' },
      });
      return response.data;
    } catch (error) {
      handleApiError(error, 'Macro Model Analysis');
    }
  },

  analyzeRatingTransition: async (previousRatings: string[], currentRatings: string[]): Promise<RatingTransitionResult> => {
    try {
      if (!previousRatings || !currentRatings) throw new Error('Previous and current ratings are required');
      if (previousRatings.length !== currentRatings.length) throw new Error('Previous and current ratings must have equal length');

      const sanitizedPrevRatings = previousRatings.map(r => r?.toString().trim() || '').filter(r => r);
      const sanitizedCurrRatings = currentRatings.map(r => r?.toString().trim() || '').filter(r => r);

      if (sanitizedPrevRatings.length === 0 || sanitizedCurrRatings.length === 0) {
        throw new Error('No valid ratings provided');
      }

      const response = await axios.post<RatingTransitionResult>(`${BASE_URL}/api/analyze/rating_transition`, {
        previous_ratings: sanitizedPrevRatings,
        current_ratings: sanitizedCurrRatings,
      });
      return response.data;
    } catch (error) {
      handleApiError(error, 'Rating Transition Analysis');
    }
  },

  analyzeLGD: async (params: AnalysisParams) => {
    try {
      const response = await axios.post(`${BASE_URL}/api/analyze/lgd`, params);
      return response.data as {
        metrics: Record<string, number>;
        plot_data: { actual_lgd: number[]; predicted_lgd: number[] };
        decile_data: { data: { actual_decile: string; values: number[] }[]; total_count: number };
        additional_data: any[];
      };
    } catch (error) {
      handleApiError(error, 'LGD Analysis');
    }
  },

  analyzeEAD: async (params: AnalysisParams) => {
    try {
      const response = await axios.post(`${BASE_URL}/api/analyze/ead`, params);
      return response.data as {
        metrics: Record<string, number>;
        plot_data: { actual_ead: number[]; predicted_ead: number[] };
        decile_data: { data: { actual_decile: string; values: number[] }[]; total_count: number };
        additional_data: any[];
      };
    } catch (error) {
      handleApiError(error, 'EAD Analysis');
    }
  },

  uploadFile: async (type: 'macro' | 'pd' | 'lgd' | 'ead' | 'pd_baseline', file: File) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${BASE_URL}/api/upload/${type}`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      return response.data;
    } catch (error) {
      handleApiError(error, 'File Upload');
    }
  },

  uploadLGD: async (file: File): Promise<LGDUploadResponse> => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post<LGDUploadResponse>(`${BASE_URL}/api/upload/lgd`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        validateStatus: status => (status >= 200 && status < 300) || status === 400,
      });

      if (response.status !== 200) {
        const errorMessage = response.data?.error || response.data?.details || response.data?.message || 'Unknown upload error';
        throw new Error(errorMessage);
      }
      return response.data;
    } catch (error) {
      handleApiError(error, 'LGD Upload');
    }
  },

  uploadMacroFile: async (file: File): Promise<any> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('description', 'Macro economic data file');

    try {
      const response = await axios.post(`${BASE_URL}/api/upload/macro`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      return response.data;
    } catch (error) {
      handleApiError(error, 'Macro File Upload');
    }
  },

  getSummary: async (): Promise<any> => {
    try {
      const response = await axios.get(`${BASE_URL}/api/summary`);
      return response.data;
    } catch (error) {
      handleApiError(error, 'Summary Fetch');
    }
  },

  getLGDDropdownOptions: async (): Promise<LGDDropdownOptionsResponse> => {
    try {
      const response = await axios.get<LGDDropdownOptionsResponse>(`${BASE_URL}/api/lgd/options`);
      return {
        portfolios: response.data.portfolios || [],
        quarters: response.data.quarters || [],
        modelNames: response.data.modelNames || [],
      };
    } catch (error) {
      console.error('Error fetching LGD dropdown options:', error);
      if (isAxiosError(error)) {
        const serverError = error.response?.data;
        const errorMessage = serverError?.error || serverError?.details || serverError?.message || error.message;
        throw new Error(`LGD Dropdown Options Fetch Failed: ${errorMessage}`);
      }
      return { quarters: [], portfolios: [], modelNames: [] }; // Fallback
    }
  },

  calculateStability: async (data?: any) => {
    try {
      const response = await axios.post(`${BASE_URL}/api/analyze/stability`, data || {}, {
        headers: { 'Content-Type': 'application/json' },
      });
      return response.data;
    } catch (error) {
      handleApiError(error, 'Stability Calculation');
    }
  },
  // Database API methods
  getAllDatasets: async () => {
    try {
      const response = await axios.get(`${BASE_URL}/api/database/datasets`);
      return response.data;
    } catch (error) {
      console.error('Error fetching datasets:', error);
      if (isAxiosError(error)) {
        const serverError = error.response?.data;
        const errorMessage = serverError?.error || serverError?.details || serverError?.message || error.message;
        throw new Error(`Datasets Fetch Failed: ${errorMessage}`);
      }
      throw error;
    }
  },

  getAllAnalysisResults: async () => {
    try {
      const response = await axios.get(`${BASE_URL}/api/database/analysis-results`);
      return response.data;
    } catch (error) {
      console.error('Error fetching analysis results:', error);
      if (isAxiosError(error)) {
        const serverError = error.response?.data;
        const errorMessage = serverError?.error || serverError?.details || serverError?.message || error.message;
        throw new Error(`Analysis Results Fetch Failed: ${errorMessage}`);
      }
      throw error;
    }
  },
  
  deleteDataset: async (datasetId: number) => {
    try {
      const response = await axios.delete(`${BASE_URL}/api/database/datasets/${datasetId}`);
      return response.data;
    } catch (error) {
      console.error('Error deleting dataset:', error);
      if (isAxiosError(error)) {
        const serverError = error.response?.data;
        const errorMessage = serverError?.error || serverError?.details || serverError?.message || error.message;
        throw new Error(`Dataset Deletion Failed: ${errorMessage}`);
      }
      throw error;
    }
  },
  
  deleteAnalysisResult: async (resultId: number) => {
    try {
      const response = await axios.delete(`${BASE_URL}/api/database/analysis-results/${resultId}`);
      return response.data;
    } catch (error) {
      console.error('Error deleting analysis result:', error);
      if (isAxiosError(error)) {
        const serverError = error.response?.data;
        const errorMessage = serverError?.error || serverError?.details || serverError?.message || error.message;
        throw new Error(`Analysis Result Deletion Failed: ${errorMessage}`);
      }
      throw error;
    }
  },

  updateDataset: async (datasetId: number, updateData: any) => {
    try {
      const response = await axios.put(`${BASE_URL}/api/database/datasets/${datasetId}`, updateData, {
        headers: { 'Content-Type': 'application/json' },
      });
      return response.data;
    } catch (error) {
      console.error('Error updating dataset:', error);
      if (isAxiosError(error)) {
        const serverError = error.response?.data;
        const errorMessage = serverError?.error || serverError?.details || serverError?.message || error.message;
        throw new Error(`Dataset Update Failed: ${errorMessage}`);
      }
      throw error;
    }
  },

  deleteDataset: async (datasetId: number) => {
    try {
      const response = await axios.delete(`${BASE_URL}/api/database/datasets/${datasetId}`);
      return response.data;
    } catch (error) {
      console.error('Error deleting dataset:', error);
      if (isAxiosError(error)) {
        const serverError = error.response?.data;
        const errorMessage = serverError?.error || serverError?.details || serverError?.message || error.message;
        throw new Error(`Dataset Delete Failed: ${errorMessage}`);
      }
      throw error;
    }
  },

  getAnalysisResult: async (resultId: number) => {
    try {
      const response = await axios.get(`${BASE_URL}/api/database/analysis-results/${resultId}`);
      return response.data;
    } catch (error) {
      console.error('Error fetching analysis result:', error);
      if (isAxiosError(error)) {
        const serverError = error.response?.data;
        const errorMessage = serverError?.error || serverError?.details || serverError?.message || error.message;
        throw new Error(`Analysis Result Fetch Failed: ${errorMessage}`);
      }
      throw error;
    }
  },

  deleteAnalysisResult: async (resultId: number) => {
    try {
      const response = await axios.delete(`${BASE_URL}/api/database/analysis-results/${resultId}`);
      return response.data;
    } catch (error) {
      console.error('Error deleting analysis result:', error);
      if (isAxiosError(error)) {
        const serverError = error.response?.data;
        const errorMessage = serverError?.error || serverError?.details || serverError?.message || error.message;
        throw new Error(`Analysis Result Delete Failed: ${errorMessage}`);
      }
      throw error;
    }
  },
};

export default api;