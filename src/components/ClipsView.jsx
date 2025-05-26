import React from 'react';
import { motion } from 'framer-motion';
import { FiDownload, FiCopy, FiShare2 } from 'react-icons/fi';

export default function ClipsView({ clips }) {
  return (
    <div className="space-y-8">
      <motion.h2
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-3xl font-bold text-gold text-center mb-12"
      >
        Your Generated Clips
      </motion.h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {clips.map((clip, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: index * 0.1 }}
            className="bg-secondary/50 backdrop-blur rounded-2xl border border-gold/10 overflow-hidden"
          >
            <div className="aspect-video bg-black relative">
              {/* Video player would go here */}
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-text-secondary">Video Player</span>
              </div>
            </div>
            
            <div className="p-6">
              <h3 className="text-lg font-semibold text-gold mb-2">{clip.title}</h3>
              <p className="text-text-secondary mb-4">{clip.description}</p>
              
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2 text-sm text-text-secondary">
                  <span>{clip.duration}</span>
                  <span>•</span>
                  <span>{clip.views} views</span>
                </div>
                
                <div className="flex items-center space-x-2">
                  <button className="p-2 hover:bg-gold/5 rounded-lg transition-colors">
                    <FiCopy className="w-5 h-5 text-gold" />
                  </button>
                  <button className="p-2 hover:bg-gold/5 rounded-lg transition-colors">
                    <FiShare2 className="w-5 h-5 text-gold" />
                  </button>
                  <button className="p-2 hover:bg-gold/5 rounded-lg transition-colors">
                    <FiDownload className="w-5 h-5 text-gold" />
                  </button>
                </div>
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}