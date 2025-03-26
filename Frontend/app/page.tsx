import Navbar from './Components/Navbar';
import Header from './Components/Header';
import FeaturesSection from './Components/FeaturesSection';
import HowItWorksSection from './Components/HowItWorksSection';
import StatisticsSection from './Components/StatisticsSection';
import TestimonialSection from './Components/TestimonialSection';
import CTASection from './Components/CTASection';
import Footer from './Components/Footer';

import {
  FaLeaf,
  FaMobileAlt,
  FaDatabase,
  FaCloudUploadAlt,
  FaMicroscope,
  FaRegLightbulb,
} from 'react-icons/fa';

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-b from-green-50 to-white dark:from-gray-900 dark:to-gray-800">
      <Navbar />
      <Header />
      <FeaturesSection />
      <HowItWorksSection />
      <StatisticsSection />
      <TestimonialSection />
      <CTASection/>
      <Footer />
    </main>
  );
}
