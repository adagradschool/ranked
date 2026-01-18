
import React from 'react';

interface QuestionsCardProps {
  title: string;
  pillLabel: string;
}

const QuestionsCard: React.FC<QuestionsCardProps> = ({ 
  title, 
  pillLabel
}) => {
  return (
    <div className="relative w-full max-w-4xl mx-auto mt-6 mb-12 group">
      {/* Decorative Background Glows */}
      <div className="absolute -top-12 left-0 w-full h-24 blur-3xl -z-10 pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-1000 bg-gradient-to-r from-indigo-50/10 via-purple-50/10 to-purple-50/5" />
      
      <div className="absolute -inset-1 blur-2xl -z-20 pointer-events-none bg-gradient-to-r from-indigo-50/10 via-purple-50/10 to-purple-50/10" />

      {/* Main Card Container */}
      <div className="relative bg-white border border-gray-100 rounded-[40px] shadow-[0_30px_60px_-15px_rgba(0,0,0,0.06)] p-10 pb-16 animate-in fade-in slide-in-from-bottom-8 duration-1000 overflow-hidden min-h-[200px]">
        
        {/* Subtle Horizontal Color Strip */}
        <div className="absolute -top-[1px] left-0 w-full h-[2px] bg-gradient-to-r from-transparent via-gray-200/50 to-transparent" />

        {/* Top Section */}
        <div className="relative flex items-center justify-between mb-8 pb-6 border-b border-gray-100">
          <div>
            <div className="flex items-baseline gap-3">
              <span className="text-4xl font-black tracking-tighter text-black">{title}</span>
            </div>
          </div>
        </div>

        {/* Blank Content Area */}
        <div className="flex items-center justify-center py-12">
          <p className="text-gray-300 font-medium italic">No questions indexed yet.</p>
        </div>
      </div>

      {/* Bottom Floating Pill Label */}
      <div className="absolute -bottom-6 left-1/2 -translate-x-1/2 px-8 py-3 bg-white border border-gray-200 rounded-2xl shadow-xl flex items-center gap-3 animate-bounce-subtle">
        <span className="text-sm font-bold text-gray-900">{pillLabel}</span>
      </div>
    </div>
  );
};

export default QuestionsCard;
