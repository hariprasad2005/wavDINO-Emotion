"""
Main entry point for wavDINO-Emotion project
Provides command-line interface for training, evaluation, and inference
"""

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(
        description='wavDINO-Emotion: Self-Supervised Audio-Visual Transformer for Emotion Recognition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on CREMA-D
  python main.py train --dataset crema --modality fusion
  
  # Evaluate model
  python main.py eval --model models/crema_d.pt --dataset crema
  
  # Cross-dataset evaluation
  python main.py cross-eval --model models/crema_d.pt
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--dataset', type=str, default='crema',
                             choices=['crema', 'ravdess', 'afew'],
                             help='Dataset to train on')
    train_parser.add_argument('--modality', type=str, default='fusion',
                             choices=['fusion', 'audio', 'visual'],
                             help='Modality to use')
    train_parser.add_argument('--batch-size', type=int, default=32)
    train_parser.add_argument('--epochs', type=int, default=50)
    train_parser.add_argument('--lr', type=float, default=3e-4)
    train_parser.add_argument('--output-dir', type=str, default='./checkpoints')
    train_parser.add_argument('--device', type=str, default='cuda')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('eval', help='Evaluate the model')
    eval_parser.add_argument('--model', type=str, required=True,
                            help='Path to model checkpoint')
    eval_parser.add_argument('--dataset', type=str, default='crema',
                            choices=['crema', 'ravdess', 'afew'])
    eval_parser.add_argument('--split', type=str, default='test',
                            choices=['train', 'val', 'test'])
    eval_parser.add_argument('--batch-size', type=int, default=32)
    eval_parser.add_argument('--output-dir', type=str, default='./results')
    
    # Cross-dataset evaluation
    cross_parser = subparsers.add_parser('cross-eval', help='Cross-dataset evaluation')
    cross_parser.add_argument('--model', type=str, required=True)
    cross_parser.add_argument('--dataset', type=str, default='crema')
    cross_parser.add_argument('--batch-size', type=int, default=32)
    cross_parser.add_argument('--output-dir', type=str, default='./results')
    
    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--model', type=str, required=True)
    infer_parser.add_argument('--audio-emb', type=str, required=True,
                             help='Path to audio embedding file (.npy)')
    infer_parser.add_argument('--visual-emb', type=str, required=True,
                             help='Path to visual embedding file (.npy)')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        from src.training.train import main as train_main
        sys.argv = ['train.py', '--dataset', args.dataset, '--modality', args.modality,
                   '--batch-size', str(args.batch_size), '--epochs', str(args.epochs),
                   '--lr', str(args.lr), '--output-dir', args.output_dir,
                   '--device', args.device]
        train_main()
    
    elif args.command == 'eval':
        from src.evaluation.evaluate import main as eval_main
        sys.argv = ['evaluate.py', '--model', args.model, '--dataset', args.dataset,
                   '--split', args.split, '--batch-size', str(args.batch_size),
                   '--output-dir', args.output_dir]
        eval_main()
    
    elif args.command == 'cross-eval':
        from src.evaluation.evaluate import main as eval_main
        sys.argv = ['evaluate.py', '--model', args.model, '--dataset', args.dataset,
                   '--batch-size', str(args.batch_size), '--output-dir', args.output_dir,
                   '--cross-dataset']
        eval_main()
    
    elif args.command == 'infer':
        import numpy as np
        from src.models.inference import ModelLoader
        
        loader = ModelLoader(args.model)
        audio_emb = np.load(args.audio_emb)
        visual_emb = np.load(args.visual_emb)
        
        emotion, confidence, probs = loader.predict(audio_emb, visual_emb)
        print(f"\nPredicted Emotion: {emotion}")
        print(f"Confidence: {confidence:.4f}")
        print(f"\nProbabilities:")
        for emotion_name, prob in probs.items():
            print(f"  {emotion_name}: {prob:.4f}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
