import React from 'react';
import { ChevronDown } from 'lucide-react';

export interface RankingItem {
  name: string;
  score: string;
  change?: string;
  date?: string;
}

interface RankingCardProps {
  title?: string;
  overallScore?: string;
  unifiedScore?: string;
  items: RankingItem[];
  pillLabel: string;
  variant?: 'blue' | 'orange';
  showHeader?: boolean;
  showDate?: boolean;
}

const RankingCard: React.FC<RankingCardProps> = ({ 
  title, 
  overallScore,
  unifiedScore,
  items, 
  pillLabel,
  variant = 'blue',
  showHeader = true,
  showDate = true
}) => {
  const isBlue = variant === 'blue';
  
  const gridLayout = showDate 
    ? "grid-cols-[0.8fr_2fr_0.8fr]" 
    : "grid-cols-[0.8fr_2fr]";
  
  return (
    <div className="relative w-full max-w-4xl mx-auto mt-6 mb-12 group">
      {/* Decorative Background Glows */}
      <div className={`absolute -top-12 left-0 w-full h-24 blur-3xl -z-10 pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-1000 ${
        isBlue ? 'bg-gradient-to-r from-[#5468ff]/10 via-[#b160ff]/10 to-[#b160ff]/5' : 'bg-gradient-to-r from-[#5468ff]/5 via-orange-100/10 to-[#b160ff]/5'
      }`} />
      
      <div className={`absolute -inset-1 blur-2xl -z-20 pointer-events-none ${
        isBlue ? 'bg-gradient-to-r from-indigo-50/20 via-purple-50/20 to-purple-50/10' : 'bg-gradient-to-r from-indigo-50/10 via-orange-50/10 to-purple-50/10'
      }`} />

      {/* Main Card Container */}
      <div className="relative bg-white border border-gray-100 rounded-[40px] shadow-[0_30px_60px_-15px_rgba(0,0,0,0.06)] p-10 pb-16 animate-in fade-in slide-in-from-bottom-8 duration-1000 overflow-hidden">
        
        {/* Subtle Horizontal Color Strip */}
        <div className={`absolute -top-[1px] left-0 w-full h-[2px] bg-gradient-to-r from-transparent to-transparent ${
          isBlue ? 'via-indigo-200/50' : 'via-orange-200/50'
        }`} />

        {/* Top Section */}
        {title && (
          <div className="relative flex items-center mb-6 pb-6 border-b border-gray-100">
            <div className="flex items-baseline gap-3">
              <span className="text-4xl font-black tracking-tighter text-black">{title}</span>
              {overallScore && (
                <span className="text-3xl font-black tracking-tighter bg-gradient-to-r from-[#5468ff] to-[#b160ff] bg-clip-text text-transparent opacity-60">
                  {overallScore}
                </span>
              )}
            </div>
          </div>
        )}

        {/* Unified Score Section */}
        {unifiedScore && (
          <div className="mb-12 mt-4">
            <div className="flex flex-col">
              <span className="text-8xl font-black tracking-tighter leading-none bg-gradient-to-r from-[#5468ff] to-[#b160ff] bg-clip-text text-transparent">
                {unifiedScore}
              </span>
            </div>
          </div>
        )}

        {/* Table Header */}
        {showHeader && (
          <div className={`grid ${gridLayout} px-4 py-3 text-sm font-bold text-gray-400 border-t border-gray-50 pt-8`}>
            <div className="flex items-center gap-1">Name <ChevronDown size={14} /></div>
            <div>Rank</div>
            {showDate && <div className="text-right">Last Updated</div>}
          </div>
        )}

        {/* Ranking Rows */}
        <div className="space-y-1">
          {items.map((item, idx) => (
            <div 
              key={idx} 
              className={`grid ${gridLayout} items-center px-4 py-5 border-b border-gray-50 last:border-0 hover:bg-gray-50/50 transition-colors rounded-xl`}
            >
              <div className="flex items-center">
                <span className="font-bold text-[18px] text-gray-900 truncate tracking-tight">
                  {item.name}
                </span>
              </div>
              
              <div className="flex items-center gap-6 pr-4">
                <div className="flex-1 h-2.5 bg-gray-100 rounded-full overflow-hidden">
                  <div 
                    className="h-full rounded-full transition-all duration-1000 bg-gradient-to-r from-[#5468ff] to-[#b160ff]"
                    style={{ width: `${item.score}%` }}
                  />
                </div>
                <div className="flex items-center gap-3 min-w-[100px] justify-end">
                  <span className="text-lg font-black tabular-nums text-gray-900 tracking-tight">
                    {item.score}
                  </span>
                  {item.change && (
                    <span className="text-sm font-bold text-emerald-500 whitespace-nowrap">
                      {item.change}
                    </span>
                  )}
                </div>
              </div>

              {showDate && (
                <div className="text-right text-sm font-bold text-gray-400 tabular-nums">
                  {item.date}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Bottom Floating Pill Label */}
      <div className="absolute -bottom-6 left-1/2 -translate-x-1/2 px-8 py-3 bg-white border border-gray-200 rounded-2xl shadow-xl flex items-center gap-3 animate-bounce-subtle">
        <span className="text-sm font-bold text-gray-900">{pillLabel}</span>
      </div>
    </div>
  );
};

export default RankingCard;