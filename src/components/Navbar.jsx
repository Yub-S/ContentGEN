import React from 'react';
import { motion } from 'framer-motion';

export default function Navbar() {
  return (
    <nav className="bg-black/50 backdrop-blur-lg border-b border-gold/10">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <motion.div 
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="flex items-center space-x-2"
          >
            <div className="w-8 h-8 bg-gradient-to-r from-gold to-gold-light rounded-lg" />
            <span className="text-xl font-bold text-gold">YouClipAI</span>
          </motion.div>
          
          <div className="flex items-center space-x-4">
            <a href="#" className="text-sm text-text-secondary hover:text-gold transition-colors">
              Documentation
            </a>
            <button className="bg-gold hover:bg-gold-light text-black px-4 py-2 rounded-lg text-sm font-medium transition-colors">
              Get Started
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
}