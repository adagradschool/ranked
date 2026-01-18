import React from 'react';
import { CheckCircle2, ArrowRight } from 'lucide-react';

export interface RecommendationItem {
  title: string;
  description: string;
}

interface RecommendationCardProps {
  title: string;
  items: RecommendationItem[];
  pillLabel: string;
}

const RecommendationCard: React.FC<RecommendationCardProps> = ({ 
  title, 
  items, 
  pillLabel
}) => {
  return (
    <div className="relative w-full max-w-4xl mx-auto mt-6 mb-12 group">
      {/* Decorative Background Glows */}
      <div className="absolute -top-12 left-0 w-full h-24 blur-3xl -z-10 pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-1000 bg-gradient-to-r from-[#5468ff]/10 via-[#b160ff]/10 to-[#b160ff]/5" />
      
      <div className="absolute -inset-1 blur-2xl -z-20 pointer-events-none bg-gradient-to-r from-indigo-50/10 via-purple-50/10 to-purple-50/10" />

      {/* Main Card Container */}
      <div className="relative bg-white border border-gray-100 rounded-[40px] shadow-[0_30px_60px_-15px_rgba(0,0,0,0.06)] p-10 pb-16 animate-in fade-in slide-in-from-bottom-8 duration-1000 overflow-hidden">
        
        {/* Subtle Horizontal Color Strip */}
        <div className="absolute -top-[1px] left-0 w-full h-[2px] bg-gradient-to-r from-transparent via-indigo-200/50 to-transparent" />

        {/* Top Section */}
        <div className="relative flex items-center justify-between mb-8 pb-6 border-b border-gray-100">
          <div>
            <div className="flex items-baseline gap-3">
              <span className="text-4xl font-black tracking-tighter text-black">{title}</span>
            </div>
          </div>
        </div>

        {/* Recommendations List */}
        <div className="space-y-4">
          {items.map((item, idx) => (
            <div 
              key={idx} 
              className="flex items-start gap-5 p-6 border border-gray-50 bg-gray-50/30 hover:bg-white hover:border-indigo-100 hover:shadow-lg hover:shadow-indigo-500/5 transition-all rounded-3xl group/item"
            >
              <div className="mt-1 flex-shrink-0">
                <div className="w-10 h-10 rounded-2xl bg-white border border-gray-100 flex items-center justify-center text-[#5468ff] group-hover/item:bg-[#5468ff] group-hover/item:text-white transition-colors shadow-sm">
                  <CheckCircle2 size={20} />
                </div>
              </div>
              
              <div className="flex-1">
                <h4 className="text-lg font-bold text-gray-900 mb-1 tracking-tight group-hover/item:text-[#5468ff] transition-colors">
                  {item.title}
                </h4>
                <p className="text-gray-500 text-sm font-medium leading-relaxed">
                  {item.description}
                </p>
              </div>

              <div className="self-center opacity-0 group-hover/item:opacity-100 transition-opacity translate-x-2 group-hover/item:translate-x-0 duration-300">
                <ArrowRight className="text-[#5468ff]" size={20} />
              </div>
            </div>
          ))}

          {/* Re-index Button Section */}
          <div className="pt-12 pb-10 flex justify-center">
            <button className="px-12 py-5 bg-black text-white rounded-[24px] font-black text-xl hover:bg-gray-800 transition-all active:scale-95 shadow-xl shadow-black/20 flex items-center justify-center group/reindex">
              Re-index
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RecommendationCard;