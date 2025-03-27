import React from 'react';
import Link from 'next/link';

const CTASection = () => {
  return (
    <section className="py-20 px-4">
      <div className="max-w-4xl mx-auto bg-gradient-to-r from-green-500 to-green-600 rounded-2xl p-10 md:p-16 text-center text-white">
        <h2 className="text-3xl md:text-4xl font-bold mb-6">
          Ready to diagnose your plant's health?
        </h2>
        <p className="text-xl mb-8">
          Start using our plant disease detection tool today and keep your
          plants healthy.
        </p>
        <Link
          href="/detect"
          className="inline-block bg-white text-green-600 font-bold py-3 px-8 rounded-full text-lg hover:bg-gray-100 transition-colors"
        >
          Detect Disease Now
        </Link>
      </div>
    </section>
  );
};

export default CTASection; 