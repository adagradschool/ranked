
import React, { useState } from 'react';
import { Search, BrainCircuit, ArrowRight, Sparkles, Wand2 } from 'lucide-react';
import { ModelMode } from '../types';

interface SearchInputProps {
  onSubmit: (query: string) => void;
  isLoading: boolean;
  modelMode: ModelMode;
  onModeChange: (mode: ModelMode) => void;
}

const SearchInput: React.FC<SearchInputProps> = ({ onSubmit, isLoading, modelMode, onModeChange }) => {
  const [value, setValue] = useState('');

  const handleSubmit = (e?: React.FormEvent) => {
    e?.preventDefault();
    if (value.trim() && !isLoading) {
      onSubmit(value);
      setValue('');
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto px-4 relative flex flex-col items-center gap-4">
      <form 
        onSubmit={handleSubmit}
        className="w-full relative flex items-center bg-white border border-gray-100 rounded-[32px] shadow-[0_4px_30px_-5px_rgba(0,0,0,0.06)] transition-all p-3 pr-4 group focus-within:border-indigo-200 focus-within:shadow-[0_8px_40px_-10px_rgba(0,0,0,0.1)]"
      >
        <div className="pl-4 pr-2 text-gray-300 group-focus-within:text-indigo-400 transition-colors">
          <Search size={22} />
        </div>
        
        <input
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Search across all AI models..."
          className="flex-1 bg-transparent border-none focus:ring-0 focus:outline-none text-xl py-4 px-2 text-black placeholder:text-gray-300 font-bold tracking-tight"
          autoFocus
        />

        <div className="flex items-center gap-3">
          <button 
            type="button"
            className={`p-3.5 bg-black text-white rounded-full transition-all flex items-center justify-center ${
              !value.trim() || isLoading 
                ? 'opacity-10 cursor-not-allowed' 
                : 'hover:bg-gray-800 scale-100 hover:scale-110 active:scale-95 shadow-xl shadow-black/10'
            }`}
            disabled={!value.trim() || isLoading}
            onClick={handleSubmit}
          >
            <ArrowRight size={24} strokeWidth={3} />
          </button>
        </div>
      </form>

      {/* Moved Expert UI Selection under the search bar */}
      <div className="flex items-center bg-gray-50/50 p-1.5 rounded-full border border-gray-100 animate-in fade-in slide-in-from-top-2 duration-700">
        {Object.values(ModelMode).map((mode) => (
          <button
            key={mode}
            onClick={() => onModeChange(mode)}
            className={`flex items-center gap-2 px-5 py-2 rounded-full text-xs font-black transition-all ${
              modelMode === mode 
                ? 'bg-white text-black shadow-sm scale-105' 
                : 'text-gray-400 hover:text-gray-600 hover:bg-gray-100/50'
            }`}
          >
            {mode === ModelMode.AUTO && <Sparkles size={14} className={modelMode === mode ? "text-amber-500" : ""} />}
            {mode === ModelMode.EXPERT && <Wand2 size={14} className={modelMode === mode ? "text-indigo-500" : ""} />}
            {mode === ModelMode.THINKING && <BrainCircuit size={14} className={modelMode === mode ? "text-purple-500" : ""} />}
            {mode}
          </button>
        ))}
      </div>
    </div>
  );
};

export default SearchInput;
