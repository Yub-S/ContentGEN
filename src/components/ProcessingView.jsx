import React from 'react';
import { motion } from 'framer-motion';

export default function ProcessingView({ onComplete }) {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh]">
      <motion.div
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        className="relative w-32 h-32"
      >
        <div className="absolute inset-0 bg-gradient-to-r from-gold to-gold-light rounded-full animate-pulse-slow" />
        <div className="absolute inset-2 bg-black rounded-full" />
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-16 h-16 border-4 border-gold border-t-transparent rounded-full animate-spin" />
        </div>
      </motion.div>
      
      <motion.h2
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mt-8 text-2xl font-semibold text-gold text-center"
      >
        Processing Your Video
      </motion.h2>
      
      <motion.p
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="mt-4 text-text-secondary text-center max-w-md"
      >
        Our AI is analyzing your content to find the perfect moments. This won't take long...
      </motion.p>
    </div>
  );
}