import React, { useState, useCallback } from 'react';
import { Upload, X, FileText, MapPin, TrendingUp, AlertCircle, Download, Play, CheckCircle2 } from 'lucide-react';

const STBGFrontend = () => {
  const [files, setFiles] = useState({});
  const [processing, setProcessing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [step, setStep] = useState('upload'); // upload, processing, results

  const requiredFiles = [
    { key: 'projects', name: 'Projects GeoJSON', description: 'Main project locations with attributes' },
    { key: 'crashes', name: 'Crashes GeoJSON', description: 'Historical crash data for safety analysis' },
    { key: 'aadt', name: 'AADT Segments', description: 'Annual Average Daily Traffic data' },
    { key: 'pop_emp', name: 'Population/Employment', description: 'TAZ data with population and employment' },
    { key: 'ej_areas', name: 'Environmental Justice Areas', description: 'EJ polygon boundaries' },
    { key: 'non_work_dest', name: 'Non-Work Destinations', description: 'Points of interest (grocery, medical, parks, etc.)' }
  ];

  const handleFileUpload = useCallback((fileKey, event) => {
    const file = event.target.files[0];
    if (file) {
      setFiles(prev => ({
        ...prev,
        [fileKey]: file
      }));
    }
  }, []);

  const removeFile = useCallback((fileKey) => {
    setFiles(prev => {
      const newFiles = { ...prev };
      delete newFiles[fileKey];
      return newFiles;
    });
  }, []);

  const processData = async () => {
    setProcessing(true);
    setError(null);
    setStep('processing');

    try {
      // Simulate processing time
      await new Promise(resolve => setTimeout(resolve, 3000));
      
      // Mock results based on your notebook output
      const mockResults = {
        projects: [
          {
            project_id: 1,
            type: 'Highway',
            county: 'Hopewell, VA',
            safety_freq: 50.0,
            safety_rate: 50.0,
            cong_demand: 10.0,
            cong_los: 0.0,
            jobs_pc: 5.0,
            jobs_pc_ej: 4.498,
            access_nw_norm: 5.0,
            access_nw_ej_norm: 4.677,
            benefit: 129.175,
            cost_mil: 14.3,
            bcr: 9.033,
            rank: 1
          },
          {
            project_id: 3,
            type: 'Intersection',
            county: 'Hopewell, VA',
            safety_freq: 3.19,
            safety_rate: 0.119,
            cong_demand: 3.377,
            cong_los: 0.0,
            jobs_pc: 0.289,
            jobs_pc_ej: 4.667,
            access_nw_norm: 2.887,
            access_nw_ej_norm: 3.464,
            benefit: 17.994,
            cost_mil: 6.7,
            bcr: 2.686,
            rank: 2
          },
          {
            project_id: 2,
            type: 'Intersection',
            county: 'Hopewell, VA',
            safety_freq: 0.0,
            safety_rate: 0.0,
            cong_demand: 0.0,
            cong_los: 0.0,
            jobs_pc: 0.283,
            jobs_pc_ej: 5.0,
            access_nw_norm: 3.89,
            access_nw_ej_norm: 5.0,
            benefit: 14.173,
            cost_mil: 5.8,
            bcr: 2.444,
            rank: 3
          }
        ]
      };

      setResults(mockResults);
      setStep('results');
    } catch (err) {
      setError('Failed to process data: ' + err.message);
      setStep('upload');
    } finally {
      setProcessing(false);
    }
  };

  const downloadResults = () => {
    const dataStr = JSON.stringify(results, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    const exportFileDefaultName = 'stbg_results.json';
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };

  const FileUploadCard = ({ fileInfo }) => {
    const hasFile = files[fileInfo.key];
    
    return (
      <div className={`border-2 border-dashed rounded-lg p-6 transition-colors ${
        hasFile ? 'border-green-300 bg-green-50' : 'border-gray-300 hover:border-gray-400'
      }`}>
        <div className="text-center">
          {hasFile ? (
            <div className="space-y-2">
              <CheckCircle2 className="mx-auto h-8 w-8 text-green-600" />
              <div className="flex items-center justify-center space-x-2">
                <span className="text-sm font-medium text-green-800">{hasFile.name}</span>
                <button
                  onClick={() => removeFile(fileInfo.key)}
                  className="text-red-600 hover:text-red-800"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            </div>
          ) : (
            <div className="space-y-2">
              <Upload className="mx-auto h-8 w-8 text-gray-400" />
              <label className="cursor-pointer">
                <span className="text-sm font-medium text-gray-900">{fileInfo.name}</span>
                <input
                  type="file"
                  className="hidden"
                  accept=".geojson,.json,.shp"
                  onChange={(e) => handleFileUpload(fileInfo.key, e)}
                />
              </label>
            </div>
          )}
          <p className="text-xs text-gray-500 mt-1">{fileInfo.description}</p>
        </div>
      </div>
    );
  };

  const ResultsTable = () => (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Rank</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Project</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Type</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Safety Score</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Congestion Score</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Equity Score</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Benefit</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Cost (M)</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">BCR</th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {results.projects.map((project) => {
            const safetyScore = project.safety_freq + project.safety_rate;
            const congestionScore = project.cong_demand + project.cong_los;
            const equityScore = project.jobs_pc + project.jobs_pc_ej + project.access_nw_norm + project.access_nw_ej_norm;
            
            return (
              <tr key={project.project_id} className={project.rank === 1 ? 'bg-yellow-50' : ''}>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                    project.rank === 1 ? 'bg-yellow-100 text-yellow-800' :
                    project.rank === 2 ? 'bg-gray-100 text-gray-800' :
                    'bg-orange-100 text-orange-800'
                  }`}>
                    #{project.rank}
                  </span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                  Project {project.project_id}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{project.type}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{safetyScore.toFixed(1)}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{congestionScore.toFixed(1)}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{equityScore.toFixed(1)}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{project.benefit.toFixed(1)}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">${project.cost_mil.toFixed(1)}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{project.bcr.toFixed(2)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">STBG Project Prioritization Tool</h1>
          <p className="text-lg text-gray-600">Surface Transportation Block Grant Program Analysis</p>
        </div>

        {/* Progress Indicator */}
        <div className="flex justify-center mb-8">
          <div className="flex items-center space-x-4">
            <div className={`flex items-center ${step === 'upload' ? 'text-blue-600' : 'text-green-600'}`}>
              <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                step === 'upload' ? 'bg-blue-100' : 'bg-green-100'
              }`}>
                {step === 'upload' ? <Upload className="w-4 h-4" /> : <CheckCircle2 className="w-4 h-4" />}
              </div>
              <span className="ml-2 text-sm font-medium">Upload Data</span>
            </div>
            <div className="w-8 h-0.5 bg-gray-300"></div>
            <div className={`flex items-center ${
              step === 'processing' ? 'text-blue-600' : 
              step === 'results' ? 'text-green-600' : 'text-gray-400'
            }`}>
              <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                step === 'processing' ? 'bg-blue-100' : 
                step === 'results' ? 'bg-green-100' : 'bg-gray-100'
              }`}>
                <TrendingUp className="w-4 h-4" />
              </div>
              <span className="ml-2 text-sm font-medium">Process</span>
            </div>
            <div className="w-8 h-0.5 bg-gray-300"></div>
            <div className={`flex items-center ${step === 'results' ? 'text-green-600' : 'text-gray-400'}`}>
              <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                step === 'results' ? 'bg-green-100' : 'bg-gray-100'
              }`}>
                <FileText className="w-4 h-4" />
              </div>
              <span className="ml-2 text-sm font-medium">Results</span>
            </div>
          </div>
        </div>

        {/* Upload Section */}
        {step === 'upload' && (
          <div className="bg-white rounded-lg shadow p-6 mb-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Upload Required Data Files</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
              {requiredFiles.map((fileInfo) => (
                <FileUploadCard key={fileInfo.key} fileInfo={fileInfo} />
              ))}
            </div>
            
            {error && (
              <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-md">
                <div className="flex">
                  <AlertCircle className="h-5 w-5 text-red-400" />
                  <div className="ml-3">
                    <p className="text-sm text-red-800">{error}</p>
                  </div>
                </div>
              </div>
            )}
            
            <div className="flex justify-center">
              <button
                onClick={processData}
                disabled={Object.keys(files).length < requiredFiles.length || processing}
                className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Play className="w-5 h-5 mr-2" />
                Run Analysis
              </button>
            </div>
          </div>
        )}

        {/* Processing Section */}
        {step === 'processing' && (
          <div className="bg-white rounded-lg shadow p-8 text-center">
            <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <h2 className="text-xl font-semibold text-gray-900 mb-2">Processing Your Data</h2>
            <p className="text-gray-600 mb-4">Analyzing safety, congestion, and equity metrics...</p>
            <div className="space-y-2 text-sm text-gray-500">
              <p>• Calculating crash frequency and severity scores</p>
              <p>• Analyzing traffic demand and congestion levels</p>
              <p>• Evaluating access to jobs and non-work destinations</p>
              <p>• Computing benefit-cost ratios</p>
            </div>
          </div>
        )}

        {/* Results Section */}
        {step === 'results' && results && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold text-gray-900">Project Prioritization Results</h2>
                <button
                  onClick={downloadResults}
                  className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500"
                >
                  <Download className="w-4 h-4 mr-2" />
                  Export Results
                </button>
              </div>
              
              <div className="mb-4 grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-blue-50 rounded-lg p-4 text-center">
                  <h3 className="text-lg font-medium text-blue-900">Total Projects</h3>
                  <p className="text-2xl font-bold text-blue-600">{results.projects.length}</p>
                </div>
                <div className="bg-green-50 rounded-lg p-4 text-center">
                  <h3 className="text-lg font-medium text-green-900">Top Ranked BCR</h3>
                  <p className="text-2xl font-bold text-green-600">{results.projects[0]?.bcr.toFixed(2)}</p>
                </div>
                <div className="bg-purple-50 rounded-lg p-4 text-center">
                  <h3 className="text-lg font-medium text-purple-900">Total Cost</h3>
                  <p className="text-2xl font-bold text-purple-600">
                    ${results.projects.reduce((sum, p) => sum + p.cost_mil, 0).toFixed(1)}M
                  </p>
                </div>
              </div>
              
              <ResultsTable />
            </div>

            <div className="flex justify-center">
              <button
                onClick={() => {
                  setStep('upload');
                  setResults(null);
                  setFiles({});
                }}
                className="inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md shadow-sm text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
              >
                Start New Analysis
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default STBGFrontend;