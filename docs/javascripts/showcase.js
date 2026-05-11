function initializeCarousel() {
  const carouselContainer = document.querySelector(".carousel-container");

  if (!carouselContainer || carouselContainer.dataset.carouselReady === "true") {
    return;
  }

  carouselContainer.dataset.carouselReady = "true";

  const carousel = carouselContainer.querySelector(".carousel");
  const slides = Array.from(carouselContainer.querySelectorAll(".carousel-slide"));
  const prevBtn = carouselContainer.querySelector(".carousel-btn-prev");
  const nextBtn = carouselContainer.querySelector(".carousel-btn-next");
  const indicatorsContainer = carouselContainer.querySelector(".carousel-indicators");

  if (!carousel || slides.length === 0 || !indicatorsContainer) {
    return;
  }

  let currentSlide = 0;

  slides.forEach((_, index) => {
    const indicator = document.createElement("button");
    indicator.type = "button";
    indicator.classList.add("carousel-indicator");
    indicator.setAttribute("aria-label", `Go to slide ${index + 1}`);
    indicator.addEventListener("click", () => goToSlide(index));
    indicatorsContainer.appendChild(indicator);
  });

  const indicators = Array.from(indicatorsContainer.querySelectorAll(".carousel-indicator"));

  function goToSlide(index) {
    slides.forEach((slide) => {
      slide.classList.remove("active", "prev", "next");
    });

    currentSlide = (index + slides.length) % slides.length;

    const prevIndex = (currentSlide - 1 + slides.length) % slides.length;
    const nextIndex = (currentSlide + 1) % slides.length;

    slides[currentSlide].classList.add("active");
    slides[prevIndex].classList.add("prev");
    slides[nextIndex].classList.add("next");

    indicators.forEach((indicator, indicatorIndex) => {
      indicator.classList.toggle("active", indicatorIndex === currentSlide);
      indicator.setAttribute("aria-current", indicatorIndex === currentSlide ? "true" : "false");
    });
  }

  function nextSlide() {
    goToSlide(currentSlide + 1);
  }

  function prevSlide() {
    goToSlide(currentSlide - 1);
  }

  if (prevBtn) {
    prevBtn.addEventListener("click", prevSlide);
  }

  if (nextBtn) {
    nextBtn.addEventListener("click", nextSlide);
  }

  document.addEventListener("keydown", (event) => {
    if (!document.querySelector(".carousel-container")) {
      return;
    }

    if (event.key === "ArrowLeft") {
      prevSlide();
    } else if (event.key === "ArrowRight") {
      nextSlide();
    }
  });

  let touchStartX = 0;
  let touchEndX = 0;

  carousel.addEventListener(
    "touchstart",
    (event) => {
      touchStartX = event.changedTouches[0].screenX;
    },
    false
  );

  carousel.addEventListener(
    "touchend",
    (event) => {
      touchEndX = event.changedTouches[0].screenX;
      handleSwipe();
    },
    false
  );

  function handleSwipe() {
    const swipeThreshold = 50;
    const diff = touchStartX - touchEndX;

    if (Math.abs(diff) <= swipeThreshold) {
      return;
    }

    if (diff > 0) {
      nextSlide();
    } else {
      prevSlide();
    }
  }

  goToSlide(0);
}

if (typeof document$ !== "undefined") {
  document$.subscribe(initializeCarousel);
} else if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initializeCarousel);
} else {
  initializeCarousel();
}
