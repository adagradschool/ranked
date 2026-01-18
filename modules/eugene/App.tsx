import React, { useState, useCallback } from 'react';
import SearchInput from './components/SearchInput';
import ChatInterface from './components/ChatInterface';
import Sidebar from './components/Sidebar';
import RankingCard, { RankingItem } from './components/RankingCard';
import QuestionsCard from './components/QuestionsCard';
import RecommendationCard, { RecommendationItem } from './components/RecommendationCard';
import { ViewMode, ModelMode, ChatSession, Message } from './types';
import { geminiService } from './services/gemini';
import { Settings, History, Search } from 'lucide-react';

const aiPlatforms: RankingItem[] = [
  { name: 'ChatGPT', score: '98.4', change: '5.2%' },
  { name: 'Gemini', score: '95.8', change: '3.1%' },
  { name: 'Perplexity', score: '97.2', change: '4.8%' },
  { name: 'Claude', score: '94.5', change: '2.7%' },
  { name: 'Grok', score: '92.1', change: '1.9%' },
];

const restaurantCompetitors: RankingItem[] = [
  { name: 'Pasta Palace', score: '94.2' },
  { name: 'Sushi Zen', score: '91.5' },
  { name: 'The Grill', score: '88.9' },
  { name: 'Burger Barn', score: '85.4' },
  { name: 'Taco Time', score: '82.1' },
];

const actionPlanItems: RecommendationItem[] = [
  { 
    title: 'High-Authority Blog Deep-Dives', 
    description: 'Publish 2,500+ word technical blogs focusing on AI performance metrics to establish domain authority.' 
  },
  { 
    title: 'Semantic Content Optimization', 
    description: 'Update existing pillar pages with LSI keywords discovered via recent search intent shifts.' 
  },
  { 
    title: 'Original Research & Case Studies', 
    description: 'Develop data-driven reports comparing real-world AI outputs to serve as high-quality backlink magnets.' 
  },
  { 
    title: 'User-Generated Insight Integration', 
    description: 'Convert common community queries into structured FAQ sections to capture long-tail voice search traffic.' 
  },
];

const App: React.FC = () => {
  const [currentView, setCurrentView] = useState<ViewMode>('search');
  const [modelMode, setModelMode] = useState<ModelMode>(ModelMode.AUTO);
  const [activeSession, setActiveSession] = useState<ChatSession | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [history, setHistory] = useState<Array<{ id: string, title: string }>>([]);
  const [showHistory, setShowHistory] = useState(false);
  const [activeSidebarSection, setActiveSidebarSection] = useState<'gpt' | 'analytics'>('gpt');

  const handleSearch = useCallback(async (query: string) => {
    setIsLoading(true);
    setCurrentView('chat');
    setActiveSidebarSection('gpt');

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: query,
      timestamp: new Date()
    };

    let updatedMessages = activeSession ? [...activeSession.messages, userMessage] : [userMessage];
    
    if (!activeSession) {
      const newSession: ChatSession = {
        id: Date.now().toString(),
        title: query.slice(0, 30) + (query.length > 30 ? '...' : ''),
        messages: updatedMessages,
        createdAt: new Date()
      };
      setActiveSession(newSession);
      setHistory(prev => [{ id: newSession.id, title: newSession.title }, ...prev]);
    } else {
      setActiveSession(prev => prev ? { ...prev, messages: updatedMessages } : null);
    }

    const result = await geminiService.performSearch(query, modelMode);
    
    const assistantMessage: Message = {
      id: (Date.now() + 1).toString(),
      role: 'assistant',
      content: result.content || "I couldn't process that.",
      sources: result.sources,
      timestamp: result.timestamp || new Date()
    };

    setActiveSession(prev => prev ? { 
      ...prev, 
      messages: [...prev.messages, assistantMessage] 
    } : null);
    
    setIsLoading(false);
  }, [activeSession, modelMode]);

  const resetSearch = () => {
    setCurrentView('search');
    setActiveSession(null);
  };

  const handleSidebarChange = (section: 'gpt' | 'analytics') => {
    setActiveSidebarSection(section);
    if (section === 'analytics') {
      setCurrentView('search');
    }
  };

  return (
    <div className="flex flex-col h-screen bg-white overflow-hidden">
      <header className="h-16 flex items-center justify-between px-6 border-b border-gray-50 flex-shrink-0 bg-white/80 backdrop-blur-md sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <div 
            className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-all" 
            onClick={resetSearch}
          >
            <h1 className="text-xl font-black tracking-tighter text-black">
              Ranked<span className="bg-gradient-to-r from-[#5468ff] to-[#b160ff] bg-clip-text text-transparent">GPT</span>
            </h1>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button 
            onClick={() => setShowHistory(!showHistory)}
            className="p-2.5 text-gray-400 hover:text-black hover:bg-gray-50 rounded-xl transition-all relative"
          >
            <History size={20} />
            {history.length > 0 && (
              <span className="absolute top-2 right-2 w-2 h-2 bg-indigo-500 rounded-full border-2 border-white" />
            )}
          </button>
          <button className="p-2.5 text-gray-400 hover:text-black hover:bg-gray-50 rounded-xl transition-all">
            <Settings size={20} />
          </button>
          <div className="h-8 w-px bg-gray-100 mx-2" />
          <div className="w-9 h-9 rounded-full bg-gradient-to-tr from-[#5468ff] to-[#b160ff] overflow-hidden cursor-pointer hover:ring-4 ring-gray-50 transition-all border border-gray-100 shadow-sm">
             <img src="https://picsum.photos/seed/user/100/100" alt="Avatar" className="w-full h-full object-cover" />
          </div>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        <Sidebar 
          activeSection={activeSidebarSection} 
          onSectionChange={handleSidebarChange} 
        />

        <main className="flex-1 flex flex-col relative overflow-y-auto overflow-x-hidden bg-white">
          <div className="flex-1 flex flex-col">
            {activeSidebarSection === 'analytics' ? (
              <div className="flex-1 flex flex-col items-center pt-8 p-8 pb-32">
                {/* Analytics Search Card */}
                <div className="relative w-full max-w-4xl mx-auto mb-20 group animate-in fade-in slide-in-from-bottom-8 duration-1000">
                  <div className="absolute -inset-1 blur-2xl -z-20 pointer-events-none bg-gradient-to-r from-indigo-50/10 via-purple-50/10 to-purple-50/10" />
                  
                  <div className="relative bg-white border border-gray-100 rounded-[40px] shadow-[0_30px_60px_-15px_rgba(0,0,0,0.06)] p-10 pb-16 overflow-hidden">
                    <div className="absolute -top-[1px] left-0 w-full h-[2px] bg-gradient-to-r from-transparent via-indigo-200/50 to-transparent" />
                    
                    <div className="relative flex items-center justify-between mb-8 pb-6 border-b border-gray-100">
                      <h3 className="text-4xl font-black tracking-tighter text-black">Search</h3>
                    </div>

                    <div className="relative group/input">
                      <div className="absolute inset-y-0 left-6 flex items-center pointer-events-none">
                        <Search className="text-gray-300 group-focus-within/input:text-[#5468ff] transition-colors" size={24} />
                      </div>
                      <input 
                        type="text"
                        placeholder="Filter by keyword, industry, or competitor..."
                        className="w-full pl-16 pr-8 py-6 bg-gray-50/30 border border-gray-100 rounded-3xl focus:bg-white focus:ring-4 focus:ring-indigo-50/50 focus:border-indigo-100 outline-none text-xl font-bold tracking-tight transition-all text-black placeholder:text-gray-300"
                      />
                    </div>
                  </div>

                  {/* Bottom Floating Pill Label */}
                  <div className="absolute -bottom-6 left-1/2 -translate-x-1/2 px-8 py-3 bg-white border border-gray-200 rounded-2xl shadow-xl flex items-center gap-3 animate-bounce-subtle">
                    <span className="text-sm font-bold text-gray-900">Rank Discovery</span>
                  </div>
                </div>
                
                <RankingCard 
                  title="Unified Rank"
                  unifiedScore="94.6"
                  items={aiPlatforms}
                  pillLabel="Performance"
                  variant="blue"
                  showDate={false}
                  showHeader={true}
                />

                {/* Competition Section */}
                <RankingCard 
                  title="Competition"
                  items={restaurantCompetitors}
                  pillLabel="Rival IQ"
                  variant="blue"
                  showDate={false}
                  showHeader={true}
                />

                {/* Questions Section */}
                <QuestionsCard 
                  title="Questions"
                  pillLabel="User Intent"
                />

                {/* Recommendation Section */}
                <RecommendationCard 
                  title="Recommendation"
                  items={actionPlanItems}
                  pillLabel="Action Plan"
                />
              </div>
            ) : (
              <div className="flex-1 flex flex-col">
                {currentView === 'search' && !activeSession ? (
                  <div className="flex-1 flex flex-col items-center pt-48 p-8">
                    <div className="mb-14 text-center animate-in fade-in zoom-in-95 duration-1000">
                      <h2 className="text-[110px] font-black tracking-tighter leading-none text-black mb-4">
                        Ranked<span className="bg-gradient-to-r from-[#5468ff] to-[#b160ff] bg-clip-text text-transparent">GPT</span>
                      </h2>
                      <p className="font-bold text-[24px] tracking-tight max-w-2xl mx-auto leading-tight text-gray-400">
                        One score. Every AI platform. Proof it works.
                      </p>
                    </div>
                    <div className="w-full max-w-4xl mx-auto mb-12">
                      <SearchInput 
                        onSubmit={handleSearch} 
                        isLoading={isLoading} 
                        modelMode={modelMode}
                        onModeChange={setModelMode}
                      />
                    </div>
                  </div>
                ) : (
                  <ChatInterface 
                    messages={activeSession?.messages || []} 
                    isLoading={isLoading} 
                    modelMode={modelMode}
                  />
                )}
              </div>
            )}
          </div>

          {currentView === 'chat' && activeSidebarSection === 'gpt' && (
            <div className="w-full p-6 bg-gradient-to-t from-white via-white/95 to-transparent sticky bottom-0 z-20">
              <div className="max-w-5xl mx-auto">
                <SearchInput 
                  onSubmit={handleSearch} 
                  isLoading={isLoading} 
                  modelMode={modelMode}
                  onModeChange={setModelMode}
                />
              </div>
            </div>
          )}
        </main>
      </div>

      {showHistory && (
        <div className="absolute top-16 right-0 w-80 h-[calc(100vh-64px)] bg-white border-l border-gray-100 shadow-2xl z-40 animate-in slide-in-from-right duration-300">
          <div className="p-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="font-bold text-gray-900">Recent Search History</h3>
              <button onClick={() => setShowHistory(false)} className="text-gray-400 hover:text-black p-1">✕</button>
            </div>
            <div className="space-y-2">
              {history.length === 0 ? (
                <p className="text-sm text-gray-400 italic py-8 text-center">No search history yet.</p>
              ) : (
                history.map(item => (
                  <button 
                    key={item.id}
                    className="w-full text-left p-4 text-sm font-medium text-gray-600 hover:bg-gray-50 rounded-2xl transition-all border border-transparent hover:border-gray-100 group flex items-center justify-between"
                  >
                    <span className="truncate pr-4">{item.title}</span>
                    <span className="text-[10px] text-gray-300 group-hover:text-gray-400">View</span>
                  </button>
                ))
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;