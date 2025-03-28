


import CTASection from './CTASection';
import FeaturesSection from './FeaturesSection';
import Footer from './Footer';
import Header from './Header';
import HowItWorksSection from './HowItWorksSection';
import Navbar from './Navbar';
import StatisticsSection from './StatisticsSection';
import TestimonialSection from './TestimonialSection';

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-green-50 to-white dark:from-gray-900 dark:to-gray-800">
      <Navbar />
      <Header/>
      <FeaturesSection/>
      <HowItWorksSection/>
      <StatisticsSection/>
      <TestimonialSection/>
      <Footer/>
    </div>
  );
}