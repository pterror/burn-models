import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'burn-image',
  description: 'Stable Diffusion inference in pure Rust with Burn',

  base: '/burn-image/',

  themeConfig: {
    nav: [
      { text: 'Guide', link: '/' },
      { text: 'Samplers', link: '/samplers' },
      { text: 'Pipelines', link: '/pipelines' },
    ],

    sidebar: [
      {
        text: 'Guide',
        items: [
          { text: 'Introduction', link: '/' },
          { text: 'Getting Started', link: '/getting-started' },
          { text: 'Architecture', link: '/architecture' },
        ]
      },
      {
        text: 'Reference',
        items: [
          { text: 'Pipelines', link: '/pipelines' },
          { text: 'Samplers', link: '/samplers' },
          { text: 'Future Architectures', link: '/future-architectures' },
        ]
      },
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/TODO/burn-image' }
    ],

    search: {
      provider: 'local'
    },
  },
})
