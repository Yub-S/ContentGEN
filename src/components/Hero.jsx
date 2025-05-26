import React from 'react';
import { motion } from 'framer-motion';
import { FiUpload } from 'react-icons/fi';

export default function Hero({ onProcess }) {
  return (
    <div className="relative py-20 overflow-hidden">
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -inset-[10px] bg-gradient-to-r from-gold/10 to-gold-light/10 blur-3xl opacity-30" />
      </div>
      
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative max-w-4xl mx-auto text-center"
      >
        <h1 className="text-5xl md:text-6xl font-bold mb-6 text-gold">
          Transform Your Content
        </h1>
        
        <p className="text-xl text-text-secondary mb-12 max-w-2xl mx-auto">
          Create engaging social media clips from your long-form videos using AI
        </p>
        
        <div className="flex flex-col items-center justify-center space-y-6">
          <div className="w-full max-w-xl p-8 bg-secondary/50 backdrop-blur rounded-2xl border border-gold/10">
            <input
              type="text"
              placeholder="Paste your YouTube URL here..."
              className="w-full bg-black/50 border border-gold/20 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-gold/50"
            />
            
            <button
              onClick={onProcess}
              className="mt-4 w-full bg-gradient-to-r from-gold to-gold-light hover:from-gold-light hover:to-gold text-black py-3 rounded-lg font-medium flex items-center justify-center space-x-2 transition-all"
            >
              <FiUpload className="w-5 h-5" />
              <span>Generate Clips</span>
            </button>
          </div>
        </div>
      </motion.div>
    </div>
  );
}