"use client";
import React from "react";
import Slider from "react-slick";
import "slick-carousel/slick/slick.css";
import "slick-carousel/slick/slick-theme.css";
import { FaQuoteRight } from "react-icons/fa";

// Slider settings for responsiveness
const sliderSettings = {
  dots: true,
  infinite: true,
  speed: 500,
  slidesToShow: 2,
  slidesToScroll: 1,
  autoplay: true,
  autoplaySpeed: 3000,
  arrows: false,
  pauseOnHover: true,
  responsive: [
    {
      breakpoint: 1024,
      settings: {
        slidesToShow: 2,
        slidesToScroll: 1,
      }
    },
    {
      breakpoint: 768,
      settings: {
        slidesToShow: 1,
        slidesToScroll: 1,
      }
    }
  ]
};

// Testimonial data
const testimonials = [
  {
    name: "Maggie Ulrey",
    location: "Denpasar",
    quote:
      "Absolutely LOVE JosieFarms! Their organic veggies are so crisp and flavorful. Knowing they're grown naturally makes me feel great about what I eat. Five stars all the way!",
    avatar: "/3.jpg",
  },
  {
    name: "Sarah Johnson",
    location: "Home Gardener",
    quote:
      "LeafSense helped me save my tomato plants! I was able to identify early blight quickly and treat it before it spread to the rest of my garden.",
    avatar: "/4.jpg",
  },
  {
    name: "Michael Chen",
    location: "Commercial Farmer",
    quote:
      "As a commercial farmer, early detection is crucial. LeafSense has become an essential tool for our operation, helping us reduce crop losses by over 30%.",
    avatar: "/2.jpg",
  },
  {
    name: "Emma Rodriguez",
    location: "Agricultural Consultant",
    quote:
      "I recommend LeafSense to all my clients. The accuracy and detailed treatment recommendations make it stand out from other plant disease detection tools.",
    avatar: "/1.png",
  },
];

const TestimonialSection: React.FC = () => {
  return (
    <section className="py-12 md:py-20 px-5 md:px-10 bg-green-50 dark:bg-gray-900 flex flex-col lg:flex-row items-center justify-between gap-8 lg:gap-16">
      {/* Left Side - Images */}
      <div className="w-full lg:w-1/2 flex flex-col md:flex-row items-center justify-center mb-10 lg:mb-0">
        <div className="w-full md:w-2/3">
          <img
            className="w-full h-[300px] md:h-[500px] object-cover "
            src="/dd.webp"
            alt="Main Image"
          />
        </div>
        <div className="w-full md:w-1/3 flex md:flex-col items-center">
          <img
            className="w-full h-[150px] md:h-[250px] object-cover "
            src="/ss.webp"
            alt="Image 1"
          />
          <img
            className="w-full h-[150px] md:h-[250px] object-cover "
            src="/ww.webp"
            alt="Image 2"
          />
        </div>
      </div>

      {/* Right Side - Testimonials */}
      <div className="w-full lg:w-1/2">
        <div className="text-left mb-8 md:mb-10">
          <h2 className="flex items-center gochi-hand-regular mb-10 md:mb-20">
            <div className="mx-2 h-0.5 w-6 bg-[#25c656]"></div>
            <span className="text-2xl md:text-2xl">Testimonies</span>
          </h2>
          <h2 className="text-2xl md:text-3xl lg:text-4xl font-bold text-gray-800 dark:text-white -mt-6 md:-mt-16">
            What Our Users Say
          </h2>
          <p className="text-gray-600 dark:text-gray-300 mt-4 md:mt-5">
            Trusted by gardeners, farmers, and agricultural professionals
            worldwide. Our platform empowers users with the tools they need to
            enhance their gardening and farming practices, ensuring healthier
            plants and more bountiful harvests.
          </p>
        </div>

        {/* Testimonial Slider */}
        <div className="max-w-full overflow-hidden">
          <Slider {...sliderSettings}>
            {testimonials.map((testimonial, index) => (
              <div key={index} className="px-2 pb-8">
                <div className="bg-white dark:bg-gray-800 p-4 md:p-6 rounded-xl shadow-lg relative w-full h-[240px] md:h-[260px] flex flex-col justify-between">
                  <div>
                    <div className="absolute bottom-6 right-6 text-green-500 text-3xl md:text-5xl opacity-80">
                      <FaQuoteRight />
                    </div>
                    <p className="text-sm md:text-base text-gray-600 dark:text-gray-300 pr-10 mt-0">
                      {testimonial.quote}
                    </p>
                  </div>

                  <div className="flex items-center mt-auto">
                    <div className="w-10 h-10 md:w-12 md:h-12 bg-gray-200 rounded-full overflow-hidden mr-3 md:mr-4">
                      <img
                        src={testimonial.avatar}
                        alt={testimonial.name}
                        width={48}
                        height={48}
                        className="object-cover"
                      />
                    </div>
                    <div>
                      <h4 className="font-bold text-gray-800 dark:text-white text-sm md:text-base">
                        {testimonial.name}
                      </h4>
                      <p className="text-xs md:text-sm text-gray-500 dark:text-gray-400">
                        {testimonial.location}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </Slider>
        </div>
      </div>
    </section>
  );
};

export default TestimonialSection;
