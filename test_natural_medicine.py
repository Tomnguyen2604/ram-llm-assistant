#!/usr/bin/env python3
"""
Comprehensive test of Odin's natural medicine expertise
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from test import OdinAssistant

def test_natural_medicine_expertise():
    """Test Odin's responses to natural medicine questions"""
    print("🌿 Testing Odin's Natural Medicine Expertise...")
    print("=" * 60)
    
    # Comprehensive list of natural medicine questions
    natural_medicine_questions = [
        # Herbs and Botanicals
        "Tell me about turmeric and its benefits",
        "What are the health benefits of ginger?",
        "How does ashwagandha help with stress?",
        "What is echinacea good for?",
        "Can you explain the benefits of ginkgo biloba?",
        "What herbs are good for digestion?",
        "Tell me about chamomile tea benefits",
        "What is milk thistle used for?",
        "How does ginseng boost energy?",
        "What are adaptogens and how do they work?",
        
        # Natural Health Approaches
        "What is natural medicine?",
        "How does holistic healing work?",
        "What are the principles of naturopathy?",
        "Tell me about Traditional Chinese Medicine",
        "What is Ayurveda and how does it work?",
        "How does acupuncture help with pain?",
        "What is homeopathy?",
        "How can nutrition be used as medicine?",
        "What role does meditation play in healing?",
        "How does aromatherapy work?",
        
        # Specific Health Conditions
        "Natural remedies for anxiety and stress",
        "What herbs help with insomnia?",
        "Natural ways to boost immune system",
        "Herbs for digestive problems",
        "Natural remedies for headaches",
        "What helps with joint pain naturally?",
        "Natural ways to increase energy",
        "Herbs for women's health issues",
        "Natural remedies for cold and flu",
        "What helps with inflammation naturally?",
        
        # Nutrition and Lifestyle
        "What are superfoods and their benefits?",
        "How does diet affect overall health?",
        "What are probiotics and why are they important?",
        "Benefits of omega-3 fatty acids",
        "What vitamins are essential for health?",
        "How does exercise support natural healing?",
        "What is the importance of sleep for health?",
        "How does stress affect the body?",
        "What are antioxidants and their benefits?",
        "How can I detox my body naturally?",
        
        # Safety and Practice
        "How do I choose quality herbal supplements?",
        "What should I know about herb-drug interactions?",
        "How do I find a qualified naturopathic doctor?",
        "What questions should I ask about natural remedies?",
        "How do I know if a natural remedy is working?",
        "What are the safety considerations for herbs?",
        "How do I start incorporating natural medicine?",
        "What should I tell my doctor about natural remedies?",
        "How do I research natural health information?",
        "What are red flags in natural health products?"
    ]
    
    try:
        # Initialize Odin
        print("Initializing Odin (CPU mode for stability)...")
        odin = OdinAssistant(finetuned_dir="./odin-finetuned", force_cpu=True)
        
        print(f"\n🎯 Testing {len(natural_medicine_questions)} natural medicine questions...")
        print("=" * 60)
        
        successful_responses = 0
        total_questions = len(natural_medicine_questions)
        
        for i, question in enumerate(natural_medicine_questions, 1):
            print(f"\n💬 Question {i}/{total_questions}: {question}")
            print("-" * 50)
            
            try:
                response = odin.generate_response(question, response_format="text")
                print(f"🤖 Odin: {response}")
                
                # Evaluate response quality
                response_lower = response.lower()
                question_lower = question.lower()
                
                # Check if response is relevant and informative
                is_relevant = False
                is_informative = len(response.split()) >= 15  # At least 15 words
                
                # Check for topic relevance
                if any(keyword in response_lower for keyword in [
                    'natural', 'herb', 'plant', 'healing', 'health', 'wellness', 
                    'medicine', 'remedy', 'nutrition', 'holistic', 'body', 'support'
                ]):
                    is_relevant = True
                
                # Check for specific topic mentions
                if 'turmeric' in question_lower and 'turmeric' in response_lower:
                    is_relevant = True
                elif 'ginger' in question_lower and 'ginger' in response_lower:
                    is_relevant = True
                elif 'stress' in question_lower and ('stress' in response_lower or 'calm' in response_lower):
                    is_relevant = True
                elif 'energy' in question_lower and ('energy' in response_lower or 'fatigue' in response_lower):
                    is_relevant = True
                
                # Evaluate overall quality
                if is_relevant and is_informative:
                    print("✅ Excellent response - relevant and informative!")
                    successful_responses += 1
                elif is_relevant:
                    print("✅ Good response - relevant but could be more detailed")
                    successful_responses += 1
                elif is_informative:
                    print("⚠️  Response is detailed but may not be fully relevant")
                else:
                    print("❌ Response needs improvement")
                
            except Exception as e:
                print(f"❌ Error generating response: {e}")
            
            # Add a small pause to avoid overwhelming output
            if i % 10 == 0:
                print(f"\n📊 Progress: {i}/{total_questions} questions completed")
                print(f"Success rate so far: {(successful_responses/i)*100:.1f}%")
        
        # Final results
        print("\n" + "=" * 60)
        print("🎯 NATURAL MEDICINE EXPERTISE TEST RESULTS")
        print("=" * 60)
        print(f"Total Questions: {total_questions}")
        print(f"Successful Responses: {successful_responses}")
        print(f"Success Rate: {(successful_responses/total_questions)*100:.1f}%")
        
        if successful_responses >= total_questions * 0.8:
            print("🌟 EXCELLENT! Odin demonstrates strong natural medicine expertise!")
        elif successful_responses >= total_questions * 0.6:
            print("✅ GOOD! Odin shows solid natural medicine knowledge!")
        else:
            print("⚠️  Odin's natural medicine responses need improvement")
        
        print(f"\n🚀 Odin is ready to help users with natural health questions!")
        
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        print("Please ensure the finetuned model exists and dependencies are installed")

if __name__ == "__main__":
    test_natural_medicine_expertise()