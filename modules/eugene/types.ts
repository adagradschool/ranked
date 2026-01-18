
export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  sources?: Array<{
    title: string;
    uri: string;
  }>;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
}

export type ViewMode = 'search' | 'chat' | 'voice' | 'imagine';

export enum ModelMode {
  AUTO = 'Auto',
  EXPERT = 'Expert',
  THINKING = 'Thinking'
}
