import React, { useState, useCallback } from 'react';
import { Card, Upload, Button, Typography, Progress, Row, Col, Image, Space } from 'antd';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { API_URLS } from '../api/config';
import toast from 'react-hot-toast';
import {
  InboxOutlined,
  UploadOutlined,
  CheckCircleOutlined,
  EyeOutlined,
  ReloadOutlined
} from '@ant-design/icons';
import { useDropzone } from 'react-dropzone';

const { Dragger } = Upload;
const { Title, Paragraph, Text } = Typography;

interface FileInfo {
  uid: string;
  name: string;
  status: 'uploading' | 'done' | 'error';
  url?: string;
  response?: any;
}

const UploadPage: React.FC = () => {
  const navigate = useNavigate();
  const [fileList, setFileList] = useState<FileInfo[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadedImage, setUploadedImage] = useState<any>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      handleUpload(file);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.webp']
    },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024, // 10MB
    multiple: false
  });

  const handleUpload = async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);

    const newFile: FileInfo = {
      uid: Date.now().toString(),
      name: file.name,
      status: 'uploading'
    };

    setFileList([newFile]);
    setUploading(true);
    setUploadProgress(0);

    try {
      console.log('开始上传图片:', file.name);

      // 创建一个带超时和进度的fetch请求
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 300000); // 5分钟超时

      // 模拟进度更新（因为fetch API不直接支持进度回调）
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev < 90) {
            const newProgress = prev + 10;
            console.log('上传进度:', newProgress + '%');
            return newProgress;
          }
          return prev;
        });
      }, 200);

      console.log('发送请求到:', API_URLS.IMAGES.UPLOAD);

      const response = await fetch(API_URLS.IMAGES.UPLOAD, {
        method: 'POST',
        body: formData,
        signal: controller.signal
      });

      clearTimeout(timeoutId);
      clearInterval(progressInterval);
      setUploadProgress(100);

      console.log('收到响应，状态:', response.status);

      if (!response.ok) {
        throw new Error(`上传失败: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();
      console.log('上传成功，结果:', result);

      // 更新文件状态
      setFileList([{
        ...newFile,
        status: 'done',
        url: result.file_url,
        response: result
      }]);

      setUploadedImage(result);
      toast.success('图片上传成功！');

    } catch (error) {
      console.error('Upload error:', error);

      let errorMessage = '未知错误';
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          errorMessage = '上传超时，请重试';
        } else {
          errorMessage = error.message;
        }
      }

      setFileList([{
        ...newFile,
        status: 'error'
      }]);
      toast.error(`图片上传失败: ${errorMessage}`);
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  };

  const handleReset = () => {
    setFileList([]);
    setUploadedImage(null);
    setUploadProgress(0);
  };

  const handleNext = () => {
    if (uploadedImage) {
      navigate('/hairstyle-selection', {
        state: { imageId: uploadedImage.image_id }
      });
    }
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        type: 'spring',
        stiffness: 100
      }
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-50">
      <motion.div
        initial="hidden"
        animate="visible"
        variants={containerVariants}
        className="container mx-auto px-4 py-8"
      >
        {/* Header */}
        <motion.div variants={itemVariants} className="text-center mb-8">
          <Title level={2} className="text-3xl font-bold mb-4">
            步骤 1/5: 上传您的照片
          </Title>
          <Paragraph className="text-gray-600">
            请上传清晰的正面照片，确保面部清晰可见，效果会更佳！
          </Paragraph>
        </motion.div>

        {/* Progress Bar */}
        <motion.div variants={itemVariants} className="mb-8">
          <Progress
            percent={20}
            strokeColor={{
              '0%': '#108ee9',
              '100%': '#87d068',
            }}
            showInfo={false}
          />
        </motion.div>

        <Row gutter={[24, 24]}>
          {/* Upload Area */}
          <Col xs={24} lg={16}>
            <motion.div variants={itemVariants}>
              <Card className="upload-card">
                {!uploadedImage ? (
                  <div
                    {...getRootProps()}
                    className={`upload-area ${isDragActive ? 'dragover' : ''}`}
                    style={{
                      border: '2px dashed #d9d9d9',
                      borderRadius: '12px',
                      padding: '60px 20px',
                      textAlign: 'center',
                      background: '#fafafa',
                      cursor: 'pointer',
                      transition: 'all 0.3s ease'
                    }}
                  >
                    <input {...getInputProps()} />
                    <InboxOutlined style={{ fontSize: '64px', color: '#1890ff', marginBottom: '16px' }} />

                    {uploading ? (
                      <div>
                        <Title level={4}>正在上传图片...</Title>
                        <Paragraph className="text-gray-500 mt-2">
                          请稍候，正在处理您的图片
                        </Paragraph>
                      </div>
                    ) : (
                      <div>
                        <Title level={4}>
                          {isDragActive ? '放开鼠标即可上传' : '点击或拖拽图片到此处上传'}
                        </Title>
                        <Paragraph className="text-gray-500">
                          支持 JPG、PNG、WebP 格式，最大 10MB
                        </Paragraph>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="uploaded-preview">
                    <div className="text-center mb-4">
                      <CheckCircleOutlined
                        style={{ fontSize: '48px', color: '#52c41a' }}
                      />
                      <Title level={4} className="mt-2">上传成功！</Title>
                    </div>

                    <div className="flex justify-center mb-6">
                      <div className="relative">
                        <Image
                          src={uploadedImage.file_url}
                          alt="上传的图片"
                          style={{ maxWidth: '400px', maxHeight: '400px' }}
                          className="rounded-lg shadow-lg"
                        />
                        {uploadedImage.face_detected && (
                          <div className="absolute top-2 right-2 bg-green-500 text-white px-2 py-1 rounded text-sm">
                            ✓ 已检测到人脸
                          </div>
                        )}
                      </div>
                    </div>

                    <div className="text-center">
                      <Space size="middle">
                        <Button
                          icon={<ReloadOutlined />}
                          onClick={handleReset}
                        >
                          重新上传
                        </Button>
                        <Button
                          type="primary"
                          icon={<EyeOutlined />}
                          onClick={handleNext}
                          size="large"
                          className="btn-gradient"
                        >
                          下一步 →
                        </Button>
                      </Space>
                    </div>
                  </div>
                )}
              </Card>
            </motion.div>
          </Col>

          {/* Tips Sidebar */}
          <Col xs={24} lg={8}>
            <motion.div variants={itemVariants}>
              <Card title="💡 上传小贴士" className="mb-4">
                <ul className="space-y-2">
                  <li>✅ 使用清晰的正面照片</li>
                  <li>✅ 确保光线充足</li>
                  <li>✅ 避免佩戴帽子或眼镜</li>
                  <li>✅ 背景尽量简洁</li>
                </ul>
              </Card>

              <motion.div variants={itemVariants}>
                <Card title="📸 示例图片">
                  <Row gutter={[8, 8]}>
                    {[1, 2, 3, 4].map((i) => (
                      <Col span={12} key={i}>
                        <div
                          className="example-thumbnail"
                          style={{
                            background: '#f0f0f0',
                            height: '80px',
                            borderRadius: '8px',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            cursor: 'pointer'
                          }}
                        >
                          <Text type="secondary">示例 {i}</Text>
                        </div>
                      </Col>
                    ))}
                  </Row>
                  <Paragraph className="text-center text-gray-500 mt-4 mb-0">
                    点击示例图片查看效果
                  </Paragraph>
                </Card>
              </motion.div>
            </motion.div>
          </Col>
        </Row>
      </motion.div>
    </div>
  );
};

export default UploadPage;