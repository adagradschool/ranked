
import React, { useRef, useEffect } from 'react';
import { Message, ModelMode } from '../types';
import { ExternalLink, User, Bot, Loader2, Sparkles, BrainCircuit } from 'lucide-react';

interface ChatInterfaceProps {
  messages: Message[];
  isLoading: boolean;
  modelMode: ModelMode;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ messages, isLoading, modelMode }) => {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  return (
    <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-8">
      <div className="max-w-3xl mx-auto space-y-10">
        {messages.map((message) => (
          <div key={message.id} className="flex gap-6 animate-in fade-in slide-in-from-bottom-2 duration-500">
            <div className="flex-shrink-0 mt-1">
              {message.role === 'user' ? (
                <div className="w-8 h-8 rounded-full bg-gray-100 flex items-center justify-center">
                  <User size={18} className="text-gray-600" />
                </div>
              ) : (
                <div className="w-8 h-8 rounded-full bg-black flex items-center justify-center">
                  <Bot size={18} className="text-white" />
                </div>
              )}
            </div>
            
            <div className="flex-1 space-y-4">
              <div className="prose prose-sm max-w-none text-gray-800 leading-relaxed text-base">
                {message.content.split('\n').map((para, i) => (
                  <p key={i} className="mb-4 last:mb-0">{para}</p>
                ))}
              </div>

              {message.sources && message.sources.length > 0 && (
                <div className="pt-4 border-t border-gray-100">
                  <h4 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-3 flex items-center gap-2">
                    <Sparkles size={14} className="text-amber-500" />
                    Sources Found
                  </h4>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                    {message.sources.map((source, idx) => (
                      <a
                        key={idx}
                        href={source.uri}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center justify-between p-3 bg-gray-50 hover:bg-gray-100 rounded-xl border border-gray-100 transition-colors group"
                      >
                        <div className="flex flex-col truncate">
                          <span className="text-sm font-medium text-gray-700 truncate">{source.title}</span>
                          <span className="text-[10px] text-gray-400 truncate">{new URL(source.uri).hostname}</span>
                        </div>
                        <ExternalLink size={14} className="text-gray-400 group-hover:text-black" />
                      </a>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex gap-6 animate-pulse">
            <div className="flex-shrink-0">
               <div className="w-8 h-8 rounded-full bg-gray-100 flex items-center justify-center">
                  <Loader2 size={18} className="text-gray-400 animate-spin" />
               </div>
            </div>
            <div className="flex-1 space-y-2 mt-2">
              <div className="h-4 bg-gray-100 rounded-full w-3/4"></div>
              <div className="h-4 bg-gray-100 rounded-full w-full"></div>
              <div className="h-4 bg-gray-100 rounded-full w-1/2"></div>
              {modelMode === ModelMode.THINKING && (
                <div className="mt-4 flex items-center gap-2 text-xs text-blue-500 font-semibold italic">
                   <BrainCircuit size={14} className="animate-pulse" />
                   Ranked is thinking deeply...
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatInterface;
