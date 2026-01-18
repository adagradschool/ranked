
import React from 'react';
import { 
  MessageSquare, 
  BarChart3
} from 'lucide-react';

interface SidebarProps {
  activeSection: 'gpt' | 'analytics';
  onSectionChange: (section: 'gpt' | 'analytics') => void;
}

const Sidebar: React.FC<SidebarProps> = ({ activeSection, onSectionChange }) => {
  return (
    <aside className="w-64 h-full border-r border-gray-100 bg-white flex flex-col flex-shrink-0 z-30 overflow-y-auto">
      <div className="p-6">
        <nav className="space-y-2">
          <button 
            onClick={() => onSectionChange('gpt')}
            className={`
              w-full flex items-center gap-4 px-5 py-4 rounded-[20px] text-base font-black transition-all duration-300
              ${activeSection === 'gpt' 
                ? 'bg-gray-900 text-white shadow-lg shadow-gray-200 scale-[1.02]' 
                : 'text-gray-400 hover:bg-gray-50 hover:text-gray-900'}
            `}
          >
            <MessageSquare size={22} strokeWidth={activeSection === 'gpt' ? 2.5 : 2} />
            <span>GPT</span>
          </button>

          <button 
            onClick={() => onSectionChange('analytics')}
            className={`
              w-full flex items-center gap-4 px-5 py-4 rounded-[20px] text-base font-black transition-all duration-300
              ${activeSection === 'analytics' 
                ? 'bg-gray-900 text-white shadow-lg shadow-gray-200 scale-[1.02]' 
                : 'text-gray-400 hover:bg-gray-50 hover:text-gray-900'}
            `}
          >
            <BarChart3 size={22} strokeWidth={activeSection === 'analytics' ? 2.5 : 2} />
            <span>Analytics</span>
          </button>
        </nav>
      </div>

      <div className="mt-auto p-6 border-t border-gray-50 opacity-20 pointer-events-none">
        {/* Footer area cleared to match minimal aesthetic */}
      </div>
    </aside>
  );
};

export default Sidebar;
