import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'burn-image',
  description: 'Stable Diffusion inference in pure Rust with Burn',

  base: '/burn-image/',

  themeConfig: {
    nav: [
      { text: 'Guide', link: '/' },
      { text: 'API', link: '/api/' },
    ],

    sidebar: [
      {
        text: 'Guide',
        items: [
          { text: 'Introduction', link: '/' },
          { text: 'Architecture', link: '/architecture' },
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
