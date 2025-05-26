import React from 'react';
import { motion } from 'framer-motion';
import { FiCpu, FiEdit3, FiShare2 } from 'react-icons/fi';

const features = [
  {
    icon: FiCpu,
    title: 'AI-Powered Analysis',
    description: 'Advanced algorithms identify the most engaging moments from your content.'
  },
  {
    icon: FiEdit3,
    title: 'Smart Captions',
    description: 'Automatically generate platform-optimized captions for maximum engagement.'
  },
  {
    icon: FiShare2,
    title: 'Ready to Share',
    description: 'Export clips perfectly formatted for all major social media platforms.'
  }
];

export default function Features() {
  return (
    <div className="py-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.2 }}
              className="relative p-6 bg-secondary/50 backdrop-blur rounded-2xl border border-gold/10"
            >
              <div className="absolute -inset-0.5 bg-gradient-to-r from-gold/10 to-gold-light/10 rounded-2xl blur opacity-75" />
              <div className="relative flex flex-col items-center text-center">
                <feature.icon className="w-8 h-8 text-gold mb-4" />
                <h3 className="text-lg font-semibold text-gold mb-2">{feature.title}</h3>
                <p className="text-text-secondary">{feature.description}</p>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
}