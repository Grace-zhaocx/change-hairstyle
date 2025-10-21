import React from 'react'
import { Routes, Route } from 'react-router-dom'
import { Layout } from 'antd'
import { Toaster } from 'react-hot-toast'
import HomePage from '@pages/HomePage'
import UploadPage from '@pages/UploadPage'
import HairstyleSelectionPage from '@pages/HairstyleSelectionPage'
import ProcessingPage from '@pages/ProcessingPage'
import PreviewPage from '@pages/PreviewPage'
import ResultPage from '@pages/ResultPage'
import Header from '@components/Layout/Header'
import Footer from '@components/Layout/Footer'
import ErrorBoundary from '@components/ErrorBoundary'
import '@styles/global.css'

const { Content } = Layout

const App: React.FC = () => {
  return (
    <ErrorBoundary>
      <Layout className="min-h-screen">
        <Header />
        <Content className="flex-1">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/hairstyle-selection" element={<HairstyleSelectionPage />} />
            <Route path="/processing" element={<ProcessingPage />} />
            <Route path="/preview" element={<PreviewPage />} />
            <Route path="/result" element={<ResultPage />} />
          </Routes>
        </Content>
        <Footer />
        <Toaster
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: '#363636',
              color: '#fff',
            },
            success: {
              duration: 3000,
              iconTheme: {
                primary: '#4aed88',
                secondary: '#fff',
              },
            },
            error: {
              duration: 5000,
              iconTheme: {
                primary: '#ff6b6b',
                secondary: '#fff',
              },
            },
          }}
        />
      </Layout>
    </ErrorBoundary>
  )
}

export default App